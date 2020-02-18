import argparse
import threading
import webbrowser
import http.server


class iAlgebraHandler(http.server.SimpleHTTPRequestHandler):
    
    def do_POST(self):
        print("got post!!")
        print("headers:", self.headers)
        content_len = int(self.headers.get('Content-Length'))
        post_body = self.rfile.read(content_len)
        print("post_body:", post_body)
        try:
            result = 'https://github.com/huashen218/i-Algebra-database/blob/master/example_adv.png?raw=true'
        except:
            result = 'error'
        self.wfile.write(bytes(result, "utf8"))
        print("result:", result)
        # self.wfile.write(result)


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
    parser.add_argument('-p', '--port', default = 8890, help="specify port")
    parser.add_argument('-f', '--html_file', default = "./ialgebra.html", help="file address of user interface")
    config=parser.parse_args()
    open_browser(config)
    start_server(config)