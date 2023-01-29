# Deep_Learning_Project
Ниже заключительный проект по курсу FW22 Deep Learning School на Stepic.


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

Весь процесс обучения представлен в [ноутбуке](https://github.com/annkka3/Deep_Learning_Project/blob/main/pix2pix-coloring-from-grayscale.ipynb)

Здесь сохранены веса обученной модели, их можно скачать по ссылке для [Monet](https://drive.google.com/file/d/1vidu5XTOFlKyQIKMJAmXnCN9ZLKR6Ano/view?usp=share_link) и для [Malevich](https://drive.google.com/file/d/11yCKZPq8m7DSqYI4CqgrSPILB8hKnizE/view?usp=share_link)

Как можно видеть из фото результат оказался неплохим. Ниже фото обучения сети на датасете Claude Monet:

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring%20from%20grayscale%20samples/Monet/samples_gen%2012.02.11.png)

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring%20from%20grayscale%20samples/Monet/samples_gen%2012.03.49.png)


![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring%20from%20grayscale%20samples/Monet/samples_gen%2014.53.52.png)

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring%20from%20grayscale%20samples/Monet/samples_gen%2014.54.38.png)


А вот обучение на датасете Kazimir Malevich:

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring%20from%20grayscale%20samples/Malevich/samples_gen%2015.02.42.png)

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring%20from%20grayscale%20samples/Malevich/samples_gen%2015.02.28.png)


![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring%20from%20grayscale%20samples/Malevich/samples_gen%2015.05.47.png)

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring%20from%20grayscale%20samples/Malevich/samples_gen%2015.05.33.png)

Больше примеров результатов можно увидеть в следующей папке Coloring from Grayscale [link](https://github.com/annkka3/Deep_Learning_Project/tree/main/Coloring%20from%20grayscale)


Но изучив документацию https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix я обратила внимание на то, что в данном случае при колоризации используется другой формат подачи входного изображения:

В цветовом пространстве L*a*b у нас как и в RGB есть три числа для каждого пикселя, но эти числа имеют разные значения. Первое число (канал), L, кодирует яркость каждого пикселя, и когда мы визуализируем этот канал, оно появляется как черно-белое изображение. Каналы *a и *b кодируют количество зелено-красного и желто-синего цвета в каждом пикселе соответственно. На следующем изображении показан каждый канал цветового пространства L*a*b отдельно.

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Additional/rgb.jpg)

Изучив документы по раскрашиванию на GitHub, люди используют цветовое пространство L*a*b вместо RGB для обучения моделей. Есть несколько причин для этого выбора:
- чтобы обучить модель раскрашиванию, мы должны дать ей изображение в градациях серого и надеяться, что модель сделает его цветным. При использовании L*a*b мы можем дать модели канал L (который представляет собой изображение в градациях серого) и предоставить модели предсказать два других канала (*a, *b), и после предсказания мы объединяем все каналы и получаем красочное изображение. Но если вы используете RGB, вы должны сначала преобразовать свое изображение в оттенки серого, передать изображение в оттенках серого модели и надеяться, что она предскажет для вас 3 канала, что является более сложной и нестабильной задачей из-за гораздо большего количества возможных комбинаций 3 каналов чисел по сравнению с двумя. Если мы предположим, что у нас есть 256 вариантов для каждого числа, предсказание трех чисел для каждого пикселя означает выбор между 256³ комбинациями, что составляет более 16 миллионов вариантов, но при прогнозировании двух чисел у нас уже около 65000 вариантов.

Для обучения используем тот же датасет [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time), в котором содержатся картины 50 самых известных художников мира.

Полный процесс обучения представлен в [ноутбуке](https://github.com/annkka3/Deep_Learning_Project/blob/main/coloring-with-pix2pix-l-ab.ipynb)

В ноутбуке в конце также имеется функция ## gen_and_print, которая генерирует из изображения (по заданному пути) и отрисовывает оригинальное и сгенерированное изображние.

Здесь сохранены веса обученной модели, их можно скачать по ссылкам для каждого из художников:

[Monet](https://drive.google.com/file/d/10NYYCtkdRf1Sb6YDZqd2U3-9Jt6UOFCt/view?usp=share_link) 

[Malevich](https://drive.google.com/file/d/1Iv4089TRYrs1H8F6QSEyAgjyTcRBzeQ4/view?usp=share_link)

[Dali](https://drive.google.com/file/d/1pr1LyJ3O-saCeb4Kb8UoUyqwY5GBGSHr/view?usp=share_link)

[Picasso](https://drive.google.com/file/d/1Mx4sYgaDsIrYh7xgDcVwWMFs3PB1d1vI/view?usp=share_link)

[Klimt](https://drive.google.com/file/d/13jqFrFrtC5iQw2iKpB9V6e5R7XFDBJ5z/view?usp=share_link)

[Warhol](https://drive.google.com/file/d/1uGscGLPnxPH7Nm0LN_uOa7mrSUh3qMsk/view?usp=share_link)


И как мы можем увидеть из полученный изображений результат действительно намного лучше:


![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring_L_ab%20samples/samples_gen%2021.53.40.png)

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring_L_ab%20samples/samples_gen%2021.54.32.png)

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring_L_ab%20samples/samples_gen%2022.05.09.png)

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring_L_ab%20samples/samples_gen%2022.06.25.png)


![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring_L_ab%20samples/samples_gen%2022.07.38.png)


![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring_L_ab%20samples/samples_gen%2022.10.39.png)

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring_L_ab%20samples/samples_gen%2022.13.16.png)

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring_L_ab%20samples/samples_gen%2022.19.42.png)

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring_L_ab%20samples/samples_gen%2022.29.08.png)


![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring_L_ab%20samples/samples_gen%2022.30.01.png)

![Иллюстрация к проекту](https://github.com/annkka3/Deep_Learning_Project/blob/main/Coloring_L_ab%20samples/samples_gen%2022.30.48.png)



Полный каталог с иллюстрациями полученными в обучении представлен ниже в папке [Coloring_L_ab samples](https://github.com/annkka3/Deep_Learning_Project/tree/main/Coloring_L_ab%20samples)



