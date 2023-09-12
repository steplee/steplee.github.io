from http.server import *
import json, numpy as np, cv2


# class Handler(BaseHTTPRequestHandler):

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kw):

        self.images = np.zeros((8,512,512,3),dtype=np.uint8)
        self.encImages = [cv2.imencode('.jpg', img)[1].tobytes() for img in self.images]
        super().__init__(*args, **kw)

    def do_GET(self):
        print(self.path)

        try:
            if self.path.startswith('/skeleton'):
                self.send_response(200)
                self.send_header('Content-Encoding', 'application/json')
                self.end_headers()
                d = {'joints': [1,2,3]}
                self.wfile.write(json.dumps(d).encode('ascii'))
            elif self.path.startswith('/image'):
                _,_,i = self.path.split('/')
                self.send_response(200)
                self.send_header('Content-Encoding', 'image/jpeg')
                self.end_headers()
                self.wfile.write(self.encImages[int(i)])
            else:
                return SimpleHTTPRequestHandler.do_GET(self)

        except:
            self.send_error(500)

        # print(self.path)
        # self.send_response(200)
        # self.end_headers()
        # msg = '<html><body><h1>Hello</h1></body></html>'
        # self.wfile.write(msg.encode('ascii'))

# def run(server_class=ThreadingHTTPServer, handler_class=Handler):
def run(server_class=HTTPServer, handler_class=Handler):
    server_address = ('', 3001)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()

run()
