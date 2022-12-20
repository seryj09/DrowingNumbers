import pygame as pg
import numpy as np
from NeuronNet import NeuralNetwork

def go_to_28():
    # Размер экрана
    size = 28 * 15
    # Размер большого пикселя в реальных пикселях
    big_size = size // 28
    # Инициализация экрана
    screen = pg.display.set_mode((size, size))
    # Установка параметров, отвечающих за частоту обновления экрана
    clock = pg.time.Clock()
    fps = 150
    # Инициаллизация массива со значением больших пикселей
    big_pix_mas_value = np.zeros((28, 28))

    # Заполнение экрана фоном
    screen.fill((0, 0, 0))
    # Условие вхождения в игровой йикл
    running = True
    # Разделитель стадий работы приложения
    not_space = True
    print_num = True
    # Начало игрового цикла
    while running:
        # Просмотр событий в окне
        events = pg.event.get()
        for event in events:
            # Непосредственно рисование цифры
            if not_space:
                # Первая часть условия для рисования линии, вторая для рисования точки
                # event.buttons[0] == 1 -- кнопка зажата
                # event.button == 1 -- кнопка была нажата
                if (event.type == pg.MOUSEMOTION and event.buttons[0] == 1) or (
                        event.type == pg.MOUSEBUTTONDOWN and event.button == 1):
                    pg.draw.circle(screen, (255, 255, 255), event.pos, 20)

                # Можно ПКМ стереть цифру
                elif (event.type == pg.MOUSEMOTION and event.buttons[2] == 1) or (
                        event.type == pg.MOUSEBUTTONDOWN and event.button == 3):
                    pg.draw.circle(screen, (0, 0, 0), event.pos, 25)

                # Просле нажатия пробела картинка преобразовывается в формат 28 * 28 пикселей
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:
                        # Вторая стадия работы, изме
                        not_space = False

                        pix_mas = pg.PixelArray(screen)

                        for i in range(28):
                            for j in range(28):
                                big_pix_mas = np.array(pix_mas[big_size * i: big_size * (i + 1),
                                                               big_size * j: big_size * (j + 1)])
                                sum_zeros = np.sum(big_pix_mas.ravel()==0)
                                white_part = round(255 * (big_size ** 2 - sum_zeros) / big_size ** 2)
                                big_pix_mas_value[i, j] = white_part


                        for i in range(28):
                            for j in range(28):
                                cur_col = [big_pix_mas_value[i, j]] * 3
                                pg.draw.rect(screen, cur_col, (big_size * i, big_size * j, big_size * (i + 1), big_size * (j + 1)))
                elif event.type == pg.QUIT:
                    running = False

            elif event.type == pg.QUIT:
                running = False
        if not not_space and print_num:
            big_pix_mas_value = big_pix_mas_value.T
            big_pix_mas_value = big_pix_mas_value.ravel()
            big_pix_mas_value = big_pix_mas_value / 255 * 0.99 + 0.01
            my_numbers = open('my numbers.txt', 'a')
            print(*big_pix_mas_value, file=my_numbers)
            my_numbers.close()

            wih_file = open('wih2.txt', 'r')
            wih = wih_file.readlines()
            wih_file.close()
            who_file = open('who2.txt', 'r')
            who = who_file.readlines()
            who_file.close()

            wih = np.array([list(map(float, i.split(' '))) for i in wih])
            who = np.array([list(map(float, i.split(' '))) for i in who])

            innodes = 28 * 28
            hnodes = 100
            onodes = 10
            learningrate = 0.1

            n = NeuralNetwork(innodes, hnodes, onodes, learningrate, wih, who)
            ans_mas = n.query(big_pix_mas_value) * 1000
            ks = open('ks.txt', 'a')
            for i, elem in enumerate(ans_mas):
                print(i, int(elem), file=ks)
            print('\n', file=ks)
            ks.close()
            print(np.argmax(ans_mas))

            print_num = False


        clock.tick(fps)
        pg.display.flip()
    pg.quit()



go_to_28()