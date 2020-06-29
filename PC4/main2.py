import face_recognition
from flask import Flask, jsonify, request, redirect, render_template
from rtree import index
from os import listdir
from os.path import isfile, isdir
import cv2
from rtree import index
import heapq
import json
from os import remove
import os.path as path

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def carpeta(cadena):
    separador = "0"
    separado_por_espacios = cadena.split(separador)
    string=separado_por_espacios[0]
    result = string.rstrip('_')
    return result+"/"

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        number=request.form['k']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename) and number:
            return detect_faces_in_image(file,number)
    return render_template("home.html")

def lista(A):
    l=[]
    for i in A:
        l.append(i)
    return l

def array_to_tupla(A):
    return tuple(A+A)

def knn_rtree(Q,k):
    if path.exists('rtree_index.data'):
        remove('rtree_index.data')
    if path.exists('rtree_index.index'):
        remove('rtree_index.index')
    p = index.Property()
    p.dimension = 256
    p.buffering_capacity = 23
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    idx = index.Index('rtree_index',properties=p)
    with open('vectors.json',errors='ignore',encoding='utf8') as contenido:
        datos=json.load(contenido)
    keys=[]
    id=0
    for i in datos:
        keys.append(i)
        idx.insert(id,array_to_tupla(datos[i]))
        id+=1

    q=array_to_tupla(Q)
    lres = list(idx.nearest(coordinates=q, num_results=k))
    print("Imagenes mas cercanas: ", lres)
    images=[]
    for i in lres:
        images.append(keys[i])
        print(keys[i])
    return images
    # return render_template("results.html")
def detect_faces_in_image(file_stream,number):
    img = face_recognition.load_image_file(file_stream)
    unknown_face_encodings = face_recognition.face_encodings(img)


    k=int(number)
    images=knn_rtree(lista(unknown_face_encodings[0]),k)
    for i in range(0,len(images)):
        images[i]=carpeta(images[i])+images[i]
        print(images[i])
    return render_template("results.html",r=images)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
