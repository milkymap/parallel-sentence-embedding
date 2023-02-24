import json 
import click 
import networkx as nx 

import torch as th 

import zmq 
import numpy as np 
import multiprocessing as mp 

from os import path 
from glob import glob 
from typing import List, Tuple, Dict, Optional, Callable, Any

from time import perf_counter

from loguru import logger 
from rich.progress import Progress, track 
from networkx.algorithms import community
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import community_detection

class TXTVectorizer:
    def __init__(self, worker_id:int, model_name:str, cache_folder:str, filepaths:List[str]):
        self.worker_id = worker_id
        print(worker_id, model_name, cache_folder)
        self.vectorizer = SentenceTransformer(
            model_name_or_path=model_name,
            cache_folder=cache_folder
        )
        self.filepaths = filepaths
        logger.debug(f'worker {self.worker_id} get {len(filepaths)} tasks and has loaded the transformers')
        
    def compute_embeddings(self) -> np.ndarray:
        if self.zmq_initialized:
            accumulator:List[np.ndarray] = []
            for path2file in self.filepaths:
                try:
                    with open(file=path2file, mode='r') as file_pointer:
                        file_content = file_pointer.read()
                        fingerprint = self.vectorizer.encode(sentences=file_content)
                        accumulator.append(fingerprint)
                        self.dealer_socket.send(b'', flags=zmq.SNDMORE)
                        self.dealer_socket.send_json({
                            'worker_id': self.worker_id,
                            'status': True,
                            'task': path2file,
                            'data': fingerprint.tolist(),
                            'error': None
                        })
                except Exception as e:
                    self.dealer_socket.send(b'', flags=zmq.SNDMORE)
                    self.dealer_socket.send_json({
                        'worker_id': self.worker_id,
                        'status': False, 
                        'task': path2file,
                        'data': None,
                        'error': e
                    })
            # end loop over files 
            return np.vstack(accumulator)
        return None 

    def __enter__(self):
        try:
            self.ctx = zmq.Context()
            self.dealer_socket:zmq.Socket = self.ctx.socket(zmq.DEALER) 
            self.dealer_socket.connect('ipc://router_dealer.ipc')
            self.zmq_initialized = True 
            logger.debug(f'worker {self.worker_id} has initialized its zeromq ressources')
        except Exception as e:
            logger.error(e)
            self.zmq_initialized = False 
        return self 

    def __exit__(self, exc, val, traceback):
        if self.zmq_initialized:
            self.dealer_socket.close(linger=0)
            self.ctx.term()
            logger.debug(f'worker {self.worker_id} has removed its zeromq ressources')
        logger.debug(f'worker {self.worker_id} has finished its work')

def vectorize_documents(worker_id:int, model_name:str, cache_folder:str, filepaths:List[str]) -> Optional[List[np.ndarray]]:
    try:
        with TXTVectorizer(worker_id, model_name, cache_folder, filepaths) as model:
            embeddings = model.compute_embeddings()
            return embeddings
    except Exception as e:
        logger.error(e)
        return None 

@click.command()
@click.option('--path2files', help='path to source files', type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option('--model_name', help='transformer model name', type=str, default='Sahajtomar/french_semantic')
@click.option('--cache_folder', help='transformer cache folder', type=str, envvar='TRANSFORMERS_CACHE', required=True)
@click.option('--nb_workers', help='number of processes(vectorizer)', type=int, default=2)
def main(path2files:str, model_name:str, cache_folder:str, nb_workers:int):
    limit = 512
    filepaths = sorted(glob(path.join(path2files, '*.txt')))[:limit]
    partitions = np.array_split(filepaths, nb_workers)
     
    ctx = zmq.Context()
    router_socket:zmq.Socket = ctx.socket(zmq.ROUTER)
    router_socket.bind('ipc://router_dealer.ipc')

    process_pool:List[mp.Process] = []
    for worker_id, partition in enumerate(partitions):
        process_ = mp.Process(
            target=vectorize_documents,
            kwargs={
                'worker_id': worker_id,
                'model_name': model_name,
                'cache_folder': cache_folder,
                'filepaths': partition
            }
        )
        process_pool.append(process_)
        process_pool[-1].start()
    
    responses = []
    try:
        with Progress() as progress:
            task_ids = [] 
            for worker_id, partition in enumerate(partitions):
                task_id = progress.add_task(description=f'worker {worker_id} embedding', total=len(partition))
                task_ids.append(task_id)

            nb_steps = 0 
            keep_loop = True
            while keep_loop:
                if router_socket.poll(timeout=100) == zmq.POLLIN:
                    _, _, encoded_msg = router_socket.recv_multipart()
                    worker_response = json.loads(encoded_msg.decode())
                    if worker_response['status']:
                        progress.update(worker_response['worker_id'], advance=1)
                        responses.append(
                            worker_response['data']
                        )
                    else:
                        logger.error(worker_response['error'])
                    nb_steps += 1 
                keep_loop = nb_steps < limit or np.any([ prs.is_alive() for prs in process_pool ]) 
            # end while loop 
        # end progress context manager
        
    except KeyboardInterrupt:
        logger.warning('ctl+c was catched') 
    except Exception as e:
        logger.error(e)
    
    for prs in process_pool:
        prs.join()

    responses = np.vstack(responses)
    print(responses, responses.shape)

    router_socket.close(linger=0)
    ctx.term()

if __name__ == '__main__':
    main()
    
    
    

