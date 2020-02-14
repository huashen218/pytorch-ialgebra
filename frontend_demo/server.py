import threading
import webbrowser
import http.server

FILE = './ialgebra.html'
PORT = 8889


class iAlgebraHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        print("got post!!")
        print("headers:", self.headers)
        content_len = int(self.headers.get('Content-Length'))
        post_body = self.rfile.read(content_len)
        print("post_body:", post_body)
        try:
            # result = post_body
            result = 'https://github.com/huashen218/i-Algebra-database/blob/master/example_adv.png?raw=true'
        except:
            result = 'error'
        self.wfile.write(bytes(result, "utf8"))
        print("result:",result)
        # self.wfile.write(result)

def open_browser():
    """Start a browser after waiting for half a second."""
    def _open_browser():
        webbrowser.open('http://lrs-twang01.ist.psu.edu:%s/%s' % (PORT, FILE))
        print(" ============== server http ==============: http://lrs-twang01.ist.psu.edu:%s/%s" % (PORT, FILE))

    thread = threading.Timer(0.5, _open_browser)
    thread.start()

def start_server():
    """Start the server."""
    server_address = ("", PORT)
    server = http.server.HTTPServer(server_address, iAlgebraHandler)
    server.serve_forever()

if __name__ == "__main__":
    print("main start....")
    open_browser()
    start_server()













# import threading
# import webbrowser
# import http.server
# import cgi

# FILE = './ialgebra.html'
# PORT = 8889

# # link:http://lrs-twang01.ist.psu.edu:8889/ialgebra.html


# class iAlgebraHandler(http.server.BaseHTTPRequestHandler):
#     def do_POST(self):
#         # # Parse the form data posted
#         # form = cgi.FieldStorage(
#         #     fp=self.rfile, 
#         #     headers=self.headers,
#         #     environ={'REQUEST_METHOD':'POST',
#         #              'CONTENT_TYPE':self.headers['Content-Type'],
#         #              })
#         # print("environ:",environ)

#         # Send response status code
#         self.send_response(200)
 
#         # Send headers
#         self.send_header('Content-type','text/html')
#         self.end_headers()
 
#         # Send message back to client
#         try:
#             # result = post_body
#             result = './img/fly_dream.jpg'
#         except:
#             result = 'error'
#         # Write content as utf-8 data
#         self.wfile.write(bytes(result, "utf8"))
#         return






# def open_browser():
#     """Start a browser after waiting for half a second."""
#     def _open_browser():
#         webbrowser.open('http://lrs-twang01.ist.psu.edu:%s/%s' % (PORT, FILE))
#         print(" ============== server http ==============: http://lrs-twang01.ist.psu.edu:%s/%s" % (PORT, FILE))

#     thread = threading.Timer(0.5, _open_browser)
#     thread.start()

# def start_server():
#     """Start the server."""
#     server_address = ("", PORT)
#     server = http.server.HTTPServer(server_address, iAlgebraHandler)
#     server.serve_forever()

# if __name__ == "__main__":
#     print("main start....")
#     open_browser()
#     start_server()










# # # from BaseHTTPServer import BaseHTTPRequestHandler
# # import http.server
# # import cgi

# # class PostHandler(http.server.BaseHTTPRequestHandler):
    
# #     def do_POST(self):
# #         # Parse the form data posted
# #         form = cgi.FieldStorage(
# #             fp=self.rfile, 
# #             headers=self.headers,
# #             environ={'REQUEST_METHOD':'POST',
# #                      'CONTENT_TYPE':self.headers['Content-Type'],
# #                      })

# #         # Begin the response
# #         self.send_response(200)
# #         self.end_headers()
# #         self.wfile.write('Client: %s\n' % str(self.client_address))
# #         self.wfile.write('User-agent: %s\n' % str(self.headers['user-agent']))
# #         self.wfile.write('Path: %s\n' % self.path)
# #         self.wfile.write('Form data:\n')

# #         # Echo back information about what was posted in the form
# #         for field in form.keys():
# #             field_item = form[field]
# #             if field_item.filename:
# #                 # The field contains an uploaded file
# #                 file_data = field_item.file.read()
# #                 file_len = len(file_data)
# #                 del file_data
# #                 self.wfile.write('\tUploaded %s as "%s" (%d bytes)\n' % \
# #                         (field, field_item.filename, file_len))
# #             else:
# #                 # Regular form value
# #                 self.wfile.write('\t%s=%s\n' % (field, form[field].value))
# #         return

# # if __name__ == '__main__':
# #     from http.server.BaseHTTPServer import HTTPServer
# #     server = HTTPServer(('localhost', 8080), PostHandler)
# #     print 'Starting server, use <Ctrl-C> to stop'
# #     server.serve_forever()