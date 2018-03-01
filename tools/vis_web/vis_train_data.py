import csv
import numpy as np


from collections import OrderedDict
from PIL import Image
import copy
import flask
from flask import Flask, render_template, request, redirect, url_for
from tools.vis_web import _get_train_stats
app = Flask(__name__)



@app.route('/', methods=['GET'])
def index():
    return render_template("index.html", ldmk_ids=info_by_ldmk.keys(), n_imgs=[len(en) for en in info_by_ldmk.itervalues()])

@app.route('/view/<string:ldmk_id>')
def view_ldmk_id(ldmk_id):

    return render_template("view.html", infos=info_by_ldmk[ldmk_id][:500], ldmk_id=ldmk_id, n_img=len(info_by_ldmk[ldmk_id]))

if __name__ == '__main__':




    info_by_ldmk = _get_train_stats()
    app.run(host='0.0.0.0', debug=True, port=5000)


