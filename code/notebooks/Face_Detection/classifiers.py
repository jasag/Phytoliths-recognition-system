# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 13:35:02 2017

@author: Jaime Sagüillo Revilla
"""

import pickle

path = '..//..//rsc//obj//'

classifiers = {}

# Todas las rutas
svm_path = path + 'svm_clf.sav'
svm2_path = path + 'svm2_clf.sav'
gtb_path = path + 'gtb_clf.sav'
rf_path = path + 'rf_clf.sav'
regr_path = path + 'rgr_clf.sav'
gnb_path = path + 'gnb_clf.sav'

# Serizalización: Cargamos clasificador e indicamos si devuelve probabilidades
classifiers['svm'] = [pickle.load(open(svm_path, 'rb')), False]
classifiers['svm2'] = [pickle.load(open(svm2_path, 'rb')), False]
classifiers['gtb'] = [pickle.load(open(gtb_path, 'rb')), True]
classifiers['rf'] = [pickle.load(open(rf_path, 'rb')), True]
classifiers['regr'] = [pickle.load(open(regr_path, 'rb')), False]
classifiers['gnb'] = [pickle.load(open(gnb_path, 'rb')), True]