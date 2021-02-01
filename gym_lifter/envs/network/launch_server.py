import socket


def launch_server():
    HOST = '147.46.89.195'
    PORT = 9999
    # SOCK_STREAM : TCP
    # AF_INET : IPv4
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # generate socket object
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # pair (host, port) : used for AF_INET
    # string representing the host name : internet domain name / IPv4 address
    # port : integer
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    # return a pair (conn, address) where conn : new socket obj usable to send & receive data
    # address : address bound to the socket on the other end of the connection
    client_socket, address = server_socket.accept()     # accept a connection
    print('Connected by ', address)
    return client_socket
