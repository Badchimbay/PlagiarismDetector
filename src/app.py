from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from src.predict_plagiarism import check_plagiarism
from src.utils.build_index import build_index, read_docx, read_pdf
import os
import torch
import platform
from pathlib import Path

ALLOWED_EXTENSIONS = {'txt', 'docx', 'pdf'}

upload_folder = os.path.join(os.getcwd(), 'temp')
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)
index_folder = os.path.join(os.getcwd(), 'data', 'raw')
if not os.path.exists(index_folder):
    os.makedirs(index_folder)

app_name = 'Антиплагиат'
app = Flask(__name__)
app.config["INDEX_FOLDER"] = index_folder
app.config["UPLOAD_FOLDER"] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def fileSort(file_location: str) -> float:
    return os.stat(file_location).st_mtime


@app.route("/")
def home():
    return render_template('home.html', app_name=app_name, title='Антиплагиат', about_text='Инфа')


@app.route("/", methods=['POST'])
def compute():
    result = []
    temp_files = []
    try:
        user_text = request.form.get('text', '').strip()
        uploaded_files = request.files.getlist('file')
        for f in uploaded_files:
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(filepath)
                temp_files.append(filepath)
                suffix = filepath.split('.')[1].lower()
                if suffix == "txt":
                    with open(filepath, 'r', encoding='utf-8') as fp:
                        user_text += '\n' + fp.read()
                elif suffix == "docx":
                    user_text += read_docx(filepath)
                elif suffix == "pdf":
                    user_text += read_pdf(filepath)
                else:
                    continue

        result_pre = check_plagiarism(user_text)
        result.append(f'{result_pre["summary"]["flagged_sentences"]} из {result_pre["summary"]["total_sentences"]} заимствованы, общий процент плагиата {result_pre["summary"]["plagiarism_percent"]}')
        for value in result_pre['details']:
            if not value['matches']:
                continue
            else:
                result.append(f'Предложение {value["sentence"]} взято из {value["matches"][0]["source"]}')
        print(result)
        if not result:
            result.append('Заимствованных предложений не найдено')
        print(result)

    finally:
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)

    result = {
        'message': '\n\n'.join(result)
    }
    return result


@app.route("/reload_indexes")
def build():
    build_index()
    return ""


@app.route("/upload", methods=['POST'])
def upload():
    uploaded_files = request.files.getlist('indexFile')
    for f in uploaded_files:
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            filepath = os.path.join(app.config['INDEX_FOLDER'], filename)
            f.save(filepath)
    return {
        'message': 'OK',
        'path': f'{filepath}'
    }


@app.route("/health")
def health():
    return "<p>YES</p>"


if __name__ == '__main__':
    app.run(debug=True)