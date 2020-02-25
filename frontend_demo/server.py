import argparse
import threading
import webbrowser
import http.server

import os
import sys
import torch
import argparse
from ialgebra.interpreters import * 
from ialgebra.utils.utils_model import load_pretrained_model
from ialgebra.utils.utils_data import load_data,loader_to_tensor
from ialgebra.utils.utils_operation import ialgebra_interpreter, save_attribution_map, vis_saliancy_map
from matplotlib.pyplot import imshow
from ialgebra.operations.operator import * 
from ialgebra.operations.compositer import * 

class iAlgebraHandler(http.server.SimpleHTTPRequestHandler):
    
    def do_POST(self):
        
        #### GET_POST ####
        content_len = int(self.headers.get('Content-Length'))
        post_body = self.rfile.read(content_len).decode("utf-8") 
        print("post_body:", post_body)


        #### PARSE_POST ####
        post_lists = post_body.split('|')
        print("post_lists:", post_lists)
        dataset, datatype, index, model_name, layer, identity_name, declarative_query = post_lists[0], post_lists[1], post_lists[2], post_lists[3], post_lists[4], post_lists[5], post_lists[6] 
        layer = int(layer[-1])
        operator_name = declarative_query


        #### FEED INTO IDLS FOR INTERPRETATION ####

            # default settings:
        batch_size = 64
        MODEL_PATH = '../checkpoints'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        SAVE_DIR = './'

            # get input
        dataloader = load_data(data_dir, dataset, datatype, batch_size = batch_size, shuffle=True)
        X_all, Y_all = loader_to_tensor(dataloader)
        bx, by = X_all[index-1: index], Y_all[index-1: index]

            # load model
        model_dir = os.path.join(MODEL_PATH, 'ckpt_' + dataset + '_' + model_name + '_' + layer + '.t7')
        model_kwargs = {'model_name': model_name, 'layer': layer, 'dataset': dataset, 'model_dir': MODEL_PATH}
        model_list = load_pretrained_model(**model_kwargs)

            # define operator
        _operator_class = Operator(identity_name, dataset, device = device)
        operator = getattr(_operator_class, operator_name)

            # get saliency map
        heatmap1, heatmapimg1 = operator(bx, by, model_list)
        save_dir = os.path.join(SAVE_DIR, dataset+'_'+index+'_'+model_name+'_'+layer+'.jpg')
        save_attribution_map(heatmap1, heatmapimg1, save_dir)

        try:
            result = save_dir
            # result = "./ialgebra_ui_demo.png"
        except:
            result = 'ERROR: NO RETURN CONTENT'

        self.wfile.write(bytes(result, "utf8"))



def open_browser(config):

    def _open_browser():
        server_address = '{}:{}/{}'.format(config.server_address, config.port, config.html_file)
        webbrowser.open(server_address)
        print("Webbrowser Address:", server_address)

    print("....Opening iAlgebra User Interface....")
    thread = threading.Timer(0.5, _open_browser)
    thread.start()



def start_server(config):

    print("....iAlgebra Server Starts Running....")
    server_address = ("", config.port)
    server = http.server.HTTPServer(server_address, iAlgebraHandler)
    server.serve_forever()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server_address', default="http://lrs-twang01.ist.psu.edu", help="specify server address")
    parser.add_argument('-p', '--port', default = 8892, help="specify port")
    parser.add_argument('-f', '--html_file', default = "./ialgebra.html", help="file address of user interface")
    config=parser.parse_args()
    open_browser(config)
    start_server(config)