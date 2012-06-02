import xmlrpclib
import sys

proxy = xmlrpclib.ServerProxy("http://localhost:8000/")
print str(proxy.recognize(sys.argv[1]))