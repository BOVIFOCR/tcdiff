
pip3 install -r requirements.txt
mkdir -p tcdiff/pretrained_models
cd tcdiff/pretrained_models
gdown --no-check-certificate -O adaface_ir50_casia.ckpt https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1AmaMTvfHq25obqb2i7MsbftJdAuvbN19
gdown --no-check-certificate -O adaface_ir50_webface4m.ckpt https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1HdW-F1GxJv0MVBUIVpE6HAZ3S9SLsytL
gdown --no-check-certificate -O center_ir_50_adaface_casia_faces_webface_112x112.pth https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1JsxekmFk-81JL9uqGR9ZUUqepo1QB53G
gdown --no-check-certificate -O center_ir_101_adaface_webface4m_faces_webface_112x112.pth https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1a6eGAl5B2hbYLdyNHD8mFhGU0YRC_bd6
gdown --no-check-certificate -O ffhq_10m_ft.pt https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1rSxpTkzO_pj_HsIfYmA6pkjXbZvHoKdm
gdown --no-check-certificate -O ffhq_10m.pt https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1rlpA4uB5GLQfUTpDVJSv22VyvHlt22xk
gdown --no-check-certificate -O simlist_ir_101_adaface_webface4m_faces_webface_112x112.pth https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1P26VhzP7jaqimrsTXNoz5cpfd2XKO-Ux

cd ../src/MICA/data
wget --no-check-certificate https://download.is.tue.mpg.de/download.php?domain=flame&resume=1&sfile=FLAME2020.zip
unzip -d FLAME2020 FLAME2020.zip
wget --no-check-certificate https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip
unzip -d FLAME2020/FLAME_masks FLAME_masks.zip
wget --no-check-certificate -P pretrained https://keeper.mpdl.mpg.de/seafhttp/files/7f78ae0d-0a72-42b2-968c-d3e35bc37dc2/mica.tar
wget --no-check-certificate -P FLAME2020 https://huggingface.co/camenduru/show/raw/main/data/head_template.obj
wget --no-check-certificate -P FLAME2020 https://huggingface.co/camenduru/show/resolve/main/data/landmark_embedding.npy

cd ../../../data
gdown --no-check-certificate -O faces_webface_112x112.zip https://drive.google.com/uc\?export\=download\&confirm\=t\&id\=1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l
unzip faces_webface_112x112.zip
