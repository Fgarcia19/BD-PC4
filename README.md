# Practica Calificada 4: Base de datos multimedia
Fabrizio Garcia Castaneda - 201810160


## Prerrequisitos
Para una mejor desarrollo del proyecto, de la carpeta que había para descargar en link propuesto, seleccione hasta un N=1000 y lo corri una vez antes de todo para guardar los vectores característicos de estas imágenes en disco y no tener que estar sacando de los vectores cada vez que lo corra lo cual consumía mucho tiempo, es por eso mismo que puse un N=1000 para que así no demore mucho al momento de extraer los vectores característicos.

## Construcción del KNN con índice Rtree
Para el índice Rtree se usó la implementación de la librería rtree, usando **from rtree import index**.

```
    p = index.Property()
    p.dimension = 256
    p.buffering_capacity = 23
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    idx = index.Index('rtree_index',properties=p)
```
A través de lo que se encuentra arriba, ponemos que la dimensión de los vectores puestos en el índice será de tamaño 256 ya que las funciones de la librería **face_recognition** devuelve un vector de dimensión de 128 y para ponerlo dentro del índice,concatenamos este vector consigo mismo para alcanzar el tamaño de 256 y que se mantenga la propiedad de las tuplas insertadas en el índice que debe tener la forma **[mina,minb,....,minz,maxa,maxb,...maxz]** y para extraer los knn se usa la función **idx.nearest(coordinates=q, num_results=k)** donde q es el vector sobre el cual se buscarán los más cercanos y k es la cantidad de elementos más cercanos que quiero retornar

## Implementación del KNN secuencial
Para este no es en sí una implementación, sino que recorre uno por uno, así sean dos completamente distintos, de inicio a fin y los va guardando en una estructura tipo heap, llamada a través de **import heapq**, en el cual voy insertando las tuplas(-score,img) hasta que llegue a una longitud de k y al llegar a esta despues de insertar se extraer el menor, como inserto con signo negativo entonces el menor sera el mas distante de esos k elementos.

## Experimentación
##### Manhattan vs Euclidiana
Se realizaron medidas de la precisión de cada una de las distancias y se obtuvieron los siguientes resultados.
Precisión| Manhattan  | Euclidiana 
-- | --| --
k=4 |4/55|4/55
k=8 | 8/55 |8/55
k=16 | 16/55 |16/55

Como se puede apreciar los resultados fueron exactamente los mismos sin embargo las imágenes que cada uno regresaba no era siempre las mismas, pero medida de **(imagenes correctas) /(total de imágenes)** se obtuvo lo mismo

##### Tiempos de ejecución

También se hicieron medidas en cuanto el tiempo de ejecución tanto de una búsqueda lineal como de una búsqueda por medio del índice R Tree y se obtuvieron los siguientes resultados
Tiempos| Secuencial  | Rtree 
-- | --| --
N=100 |0.0469|0.000637
N=200 | 0.0627 | 0.001077
N=400 | 0.0771 |0.00132
N=800 | 0.1184 |0.00259

Como se aprecia en los resultados obtenidos una búsqueda a través del R Tree es mucho mas rapida que una secuencial, debido a que no tiene que buscar por la totalidad de elementos sino en grupos cada vez más reducidos.

## Comentarios
En el uso del R Tree,este se guarda en disco por su cuenta.Sin embargo, cuando trataba de usarlo después de haber sido creado previamente me daba error siempre por lo que opte por,lo cual se podrá apreciar en el código,que cada vez que se llame a la función eliminará al índice que esté en disco y creará uno nuevo para que no me de dicho error, cabe decir que esta creación no se toma en cuenta al momento de medir los tiempos de ejecución de la búsqueda en este índice.

En main.py se encuentra la implementación para ver en consola los resultados, en main2.py te muetra los resultados en una página html, la cual es bastante simple y sin diseño ya que no he trabajado mucho diseños antes
