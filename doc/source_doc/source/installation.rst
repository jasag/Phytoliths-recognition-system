Bienvenidos a la documentación de Phytoliths-recognition-system !
=========================================================

Debemos de seguir los dos siguientes pasos para para poder ejecutar las aplicaciones:

1. Instalación de Ipython File Upload
-------------------------------------
`Ipython File Upload <https://github.com/peteut/ipython-file-upload/>`_
es la extensión que nos permitirá subir imágenes al etiquetador. Para instalarlo con pip, seguimos los siguientes pasos::

pip install --user fileupload
jupyter nbextension install --py --user fileupload
jupyter nbextension enable --py --user fileupload

2. Instalación de Jupyter Dashboards
-------------------------------------
`Jupyter Dashboards <https://github.com/jupyter/dashboards/>`_
es la extensión que nos permitirá mostrar el etiquetador con una mejor experiencia de usuario. Para instalarlo con pip, seguimos los siguientes pasos::

pip install --user jupyter_dashboards
jupyter dashboards quick-setup --user
jupyter nbextension enable jupyter_dashboards --py
