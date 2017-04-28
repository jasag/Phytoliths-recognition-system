<p align="center">
  <h1 align="center">Phytoliths recognition system</h1>
  <p align="center">
  Este proyecto contiene un sistema de reconocimiento automático de fitolitos y un etiquetador de los mismos.
  </p>
</p>
<br>

## Prerequisitos
Los prerequisitos necesarios para poder instalar las aplicaciones realizadas en este proyecto son:

- Windows.
- Google Chrome.
- Anaconda.
- La ultima *release* de este proyecto o repositorio.

## Instalación
Debemos de seguir los dos siguientes pasos para para poder ejecutar las aplicaciones:

#### 1. Instalación de Ipython File Upload
[Ipython File Upload](https://github.com/peteut/ipython-file-upload) es la extensión que nos permitirá subir imágenes al etiquetador. Para instalarlo con pip, seguimos los siguientes pasos:
```
pip install --user fileupload
jupyter nbextension install --py --user fileupload
jupyter nbextension enable --py --user fileupload
```

#### 2. Instalación de Jupyter Dashboards
[Jupyter Dashboards](https://github.com/jupyter/dashboards) es la extensión que nos permitirá mostrar el etiquetador con una mejor experiencia de usuario. Para instalarlo con pip, seguimos los siguientes pasos:

```
pip install --user jupyter_dashboards
jupyter dashboards quick-setup --user
jupyter nbextension enable jupyter_dashboards --py
```

## Uso del etiquetador
Una vez completados satisfactoriamente los pasos anteriores, ejecutamos la aplicación _Jupyter Notebook_.  Y desde esta aplicación, abrimos el notebook _Image_Labeler.ipynb_, en la carpeta _code/notebooks_ dentro de la carpeta de este proyecto. Con el _notebook_ ya abierto, tendremos que navegar por la barra de navegación de este para llevar a cabo los dos siguientes pasos:

1.  Ejecutar todas las celdas del *notebook* *Image_Labeler.ipynb* en la carpeta *code/notebooks*.  Para ello, navegamos por *Cell* y clicamos en *Run All*. 

2. Activar *Dashboard Preview*. Para ello, navegamos por *View* y clicamos en *Dashboard Preview*.

**Ya tenemos listo el etiquetador de fitolitos para su funcionamiento.**

![](https://raw.githubusercontent.com/jasag/Phytoliths-recognition-system/research/doc/img/etiquetador_de_imagenes_2_v1.JPG)


## Documentation
Para obtener información más detallada sobre el proyecto, véase la memoria y anexos del proyecto. 

## Autores
- Jaime Sagüillo Revilla

Tutores:
 - Álvar Arnaiz González
 - José Francisco Diez Pastor
 
 
## Licencia
BSD 3-Clause License

Copyright (c) 2017, Jaime Sagüillo
All rights reserved.
