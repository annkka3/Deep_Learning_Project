# Deep_Learning_Project
This is the final project of the FW22 Deep Learning School on Stepic


Для своего проекта для выбрала обучение GAN Pix2Pix.

Тренировку модели начала с датасета Maps, который содержит снимки улиц из космоса и их карточный вид.

## Maps dataset


Сам датасет можно скачать по ссылке из Kaggle: [link](https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset).

Процесс обучения и модель представлены в [ноутбуке](maps-training-pix2pix.ipynb)

Так же в репозитории отдельно сохранены модели [generator](https://github.com/annkka3/Deep_Learning_Project/blob/main/generator.py) и [discriminator](https://github.com/annkka3/Deep_Learning_Project/blob/main/discriminator.py) Pix2Pix.

Здесь сохранены веса обученной модели, их можно скачать по [ссылке.](https://drive.google.com/file/d/1N0DR8rL3Y8abHb2R4SZbfJrUny218PGW/view?usp=share_link)

Примеры результатов можно увидеть в следующей папке Maps_Samples_Generated [link](https://github.com/annkka3/Deep_Learning_Project/blob/main/Maps_Samples_Generated/readme.md)

## Примеры

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Maps_Samples_Generated/sample_transformation18.08.55.png)

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Maps_Samples_Generated/sample_transformation18.10.03.png)

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Maps_Samples_Generated/sample_transformation18.22.25.png)

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Maps_Samples_Generated/sample_transformation18.45.17.png)


## Coloring Dataset

Далее я решила обучить модель раскрашивать картины известных художников.

Мной был выбран датасет [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time), в котором содержатся картины 50 самых известных художников мира.

Сначала я решила подавать на вход модели изображения, которые преобразовывала в оттенки серого с помощью предобработки изображения  при загрузке, а затем сконкатенированное изображение серого и сгенерированного изображения, а также серого и реального изображения подавалось на вход Дискриминатору.

Как можно видеть из фото результат оказался неплохим.


