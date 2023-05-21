import json
import argparse
import os
from filelock import SoftFileLock
import netifaces as ni
import requests
import time

_COORDINATOR_CLIENT = None


def define_nccl_port_by_job_id(job_id: int):
    return 10000 + job_id % 3571  # make sure different job use different port

class CoordinatorInferenceHTTPClient:
    def __init__(self, args, model_name=None) -> None:
        self.job_id = args.job_id
        self.model_name = model_name

    def notify_inference_heartbeat(self):
        pass

    def notify_inference_join(self, netname='access', n_gpu_per_node=None):
        ip = ni.ifaddresses(netname)[ni.AF_INET][0]['addr']
        if n_gpu_per_node is None:
            return requests.post("http://173.82.206.98:5000/rank/"+str(self.job_id),
                                 json={"ip": ip}).json()
        else:
            return requests.post("http://173.82.206.98:5000/rank/"+str(self.job_id),
                                 json={"ip": ip, "n_gpu_per_node": n_gpu_per_node}).json()


def get_coordinator_client() -> CoordinatorInferenceHTTPClient:
    assert _COORDINATOR_CLIENT is not None
    return _COORDINATOR_CLIENT


def init_coordinator_client(args, model_name: str):
    global _COORDINATOR_CLIENT
    _COORDINATOR_CLIENT = CoordinatorInferenceHTTPClient(args, model_name)