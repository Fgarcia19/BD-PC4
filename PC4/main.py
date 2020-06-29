from rtree import index
from os import listdir
from os.path import isfile, isdir
import cv2
from rtree import index
import face_recognition
import heapq
import json
from os import remove
import os.path as path
from time import time

#Funciones de distancia#
def manhattan(A,B):
    return sum([abs(x1 - x2) for (x1, x2) in zip(A, B)])

def euclidiana(A,B):
    return (sum([(x1 - x2)**2 for (x1, x2) in zip(A, B)]))**(0.5)

##Sirve para convertir lo que devuelve face_recognition en una lista
def lista(A):
    l=[]
    for i in A:
        l.append(i)
    return l

##Funcion usada antes de todo para crear los vectores caracteristicos de las imagenes y guardarlos
def create_vectors():
    i=0
    path="./images/"
    vectores={}
    for car in listdir(path):
        for obj in listdir(path+car+'/'):
            print(obj)
            print(i)
            i+=1
            aux = face_recognition.load_image_file(path+car+'/'+obj)
            aux_encoding = face_recognition.face_encodings(aux)
            if len(aux_encoding)>0:
                vectores[obj]=list(aux_encoding[0])
    with open('vectors.json', 'w') as file:
        json.dump(vectores, file)

##Convertir un array en una tupla de la forma correcta para el Btree
def array_to_tupla(A):
    return tuple(A+A)

def knn_rtree(Q,k):

    #Eliminacion del indice previo para crear uno nuevo
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
    tiempo_inicial = time()
    ##Se realiza la busqueda en el indice Rtree
    q=array_to_tupla(Q)
    lres = list(idx.nearest(coordinates=q, num_results=k))
    tiempo_final = time()

    tiempo= tiempo_final - tiempo_inicial

    print("TIEMPO DE RTREE ",tiempo)

    print("Imagenes mas cercanas: ", lres)
    for i in lres:
        print(keys[i])


def knn_secuencial(q,k):
    ##Busqueda SECUENCIAL tomando las 2 distancias pedidas y al final mostrar ambos resultados
    tiempo_inicial = time()
    with open('vectors.json',errors='ignore',encoding='utf8') as contenido:
        datos=json.load(contenido)
    Q=lista(q)
    d_man=[]
    d_euc=[]
    id=0
    for i in datos:
        e=euclidiana(Q,datos[i])
        m=manhattan(Q,datos[i])
        heapq.heappush(d_man,(-m,i))
        heapq.heappush(d_euc,(-e,i))
        if len(d_man)>k:
                heapq.heappop(d_man)
        if len(d_euc)>k:
                heapq.heappop(d_euc)
    tiempo_final = time()
    tiempo= tiempo_final - tiempo_inicial

    print("manhattan")
    for i in d_man:
        print(i[0],end=' ')
        print(i[1])
    print("euclidiana")
    for i in d_euc:
        print(i[0],end=' ')
        print(i[1])
    print("TIEMPO DE SECUENCIAL ",tiempo)


##Sacar la carpeta de una foto para poder acceder a ella, ya que las fotos tienen el formato "carpeta"_0001.jpg por ejemplo
def carpeta(cadena):
    separador = "0"
    separado_por_espacios = cadena.split(separador)
    string=separado_por_espacios[0]
    result = string.rstrip('_')
    return result+"/"


##Imagen para la prueba
aux = face_recognition.load_image_file('./images/Jean_Chretien/Jean_Chretien_0020.jpg')
aux_1=face_recognition.face_encodings(aux)[0]
knn_secuencial(lista(aux_1),35)
knn_rtree(lista(aux_1),35)
