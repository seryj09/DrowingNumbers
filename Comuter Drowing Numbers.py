# В данном скрипте я попытаюсь обучать нейронку с каждым нарисованным рисунком, так как обучающие данные MNIST
# малоэффективно данные работают на цифрах, нарисованных на компьютере

import pygame as pg
import numpy as np
from NeuronNet import NeuralNetwork



# Инициализация нейронки, будет использовать веса из файла, если они есть
innodes = 28 * 28
hnodes = 100
onodes = 10
learningrate = 0.16
file_weights = True

if file_weights:
    wih_file = open('real wih.txt', 'r')
    wih = wih_file.readlines()
    wih = np.array([list(map(float, i.split(' '))) for i in wih])
    wih_file.close()

    who_file = open('real who.txt', 'r')
    who = who_file.readlines()
    who = np.array([list(map(float, i.split(' '))) for i in who])
    who_file.close()

    n = NeuralNetwork(innodes, hnodes, onodes, learningrate, wih, who)
else:
    n = NeuralNetwork(innodes, hnodes, onodes, learningrate)


def write_weights(wih, who):
    with open('real wih.txt', 'w') as f:
        for i in wih:
            print(*i, file=f)
        print('wih перезаписан')

    with open('real who.txt', 'w') as f:
        for i in who:
            print(*i, file=f)
        print('who перезаписан')



def what_number():
    real_data = open('real train.txt', 'r')
    examples = len(real_data.readlines())
    real_data.close()

    pg.init()
    # Размер экрана
    size = 28 * 15
    # Размер большого пикселя в реальных пикселях
    big_size = size // 28
    # Инициализация экрана
    screen = pg.display.set_mode((size + 400, size))
    # Инициализация области рисования
    draw_screen = pg.Surface((size, size))
    # Установка параметров, отвечающих за частоту обновления экрана
    clock = pg.time.Clock()
    fps = 150

    def print_text(message, x, y, font_color=(0, 0, 0), font_size=65, font_type='kacstbook'):
        this_font = pg.font.SysFont(font_type, font_size)
        text = this_font.render(message, 0, font_color)
        screen.blit(text, (x, y))


    # Инициаллизация массива со значением больших пикселей
    big_pix_mas_value = np.zeros((28, 28))
    # Заполнение экрана фоном
    screen.fill((0, 0, 0))
    # Нарисовали разделитель областей
    pg.draw.rect(screen, (255, 0, 0), (420, 0, 42, 420))
    # Условие вхождения в игровой цикл
    running = True
    # Разделитель стадий работы приложения
    stage = 1
    # Начало игрового цикла
    while running:
        # Просмотр событий в окне
        events = pg.event.get()
        for event in events:
            # Непосредственно рисование цифры
            if stage == 1:
                # Печать правил
                print_text('Нарисуйте слева', 470, 5, (0, 146, 230), 60)
                print_text('цифру от 0 до 9.', 470, 65, (0, 146, 230), 60)
                print_text('Затем нажмите', 490, 125, (0, 146, 230), 60)
                print_text('клавишу t.', 530, 185, (0, 146, 230), 60)
                print_text('После этого', 510, 245, (0, 146, 230), 60)
                print_text('введите вашу', 500, 305, (0, 146, 230), 60)
                print_text('цифру', 570, 365, (0, 146, 230), 60)

                # Первая часть условия для рисования линии, вторая для рисования точки
                # event.buttons[0] == 1 -- кнопка зажата
                # event.button == 1 -- кнопка была нажата
                if (event.type == pg.MOUSEMOTION and event.buttons[0] == 1) or (
                        event.type == pg.MOUSEBUTTONDOWN and event.button == 1):
                    pg.draw.circle(draw_screen, (255, 255, 255), event.pos, 20)

                # Можно ПКМ стереть цифру
                elif (event.type == pg.MOUSEMOTION and event.buttons[2] == 1) or (
                        event.type == pg.MOUSEBUTTONDOWN and event.button == 3):
                    pg.draw.circle(draw_screen, (0, 0, 0), event.pos, 25)

                # На этом этапе картинка преобразовывается в формат 28 * 28 пикселей
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_t:
                        # Вторая стадия работы, преобразование картинки и получение массива значений
                        pix_mas = pg.PixelArray(draw_screen)
                        # При первом использовании этого массива, мы сделали его одномерным, надо это исправить
                        big_pix_mas_value.resize((28, 28))
                        for i in range(28):
                            for j in range(28):
                                big_pix_mas = np.array(pix_mas[big_size * i: big_size * (i + 1),
                                                               big_size * j: big_size * (j + 1)])
                                sum_zeros = np.sum(big_pix_mas.ravel()==0)
                                white_part = round(255 * (big_size ** 2 - sum_zeros) / big_size ** 2)
                                big_pix_mas_value[i, j] = white_part
                        del pix_mas

                        for i in range(28):
                            for j in range(28):
                                cur_col = [big_pix_mas_value[i, j]] * 3
                                pg.draw.rect(draw_screen, cur_col, (big_size * i, big_size * j,
                                                                    big_size * (i + 1), big_size * (j + 1)))
                        stage = 2

                elif event.type == pg.QUIT:
                    running = False

            # Стадия обучения и запись картинки в базу
            elif stage == 3 and event.type == pg.KEYDOWN:
                if 48 <= event.key <= 57:
                    # Заполнили правую область чёрным
                    screen.fill((0, 0, 0))
                    # Нарисовали разделитель областей
                    pg.draw.rect(screen, (255, 0, 0), (420, 0, 42, 420))

                    right_answer = event.key - 48
                    target = [0.01] * 10
                    target[right_answer] = 0.99
                    n.train(input_mas, target)
                    train_data = open('real train.txt',  'a')
                    print(right_answer, *input_mas, file=train_data)
                    train_data.close()
                    stage = 1
                    draw_screen.fill((0, 0, 0))

                    examples += 1
                    print(examples)
                # Backspace
                elif event.key == 8:
                    # Заполнили правую область чёрным
                    screen.fill((0, 0, 0))
                    # Нарисовали разделитель областей
                    pg.draw.rect(screen, (255, 0, 0), (420, 0, 42, 420))

                    stage = 1
                    draw_screen.fill((0, 0, 0))



            elif event.type == pg.QUIT:
                running = False

        # Рассчёт ответа
        if stage == 2:
            big_pix_mas_value = big_pix_mas_value.T
            big_pix_mas_value = big_pix_mas_value.ravel() # Записать этот массив в файл
            input_mas = big_pix_mas_value / 255 * 0.99 + 0.01
            output_mas = n.query(input_mas)
            output_mas = [[i, elem] for i, elem in enumerate(output_mas)]
            output_mas.sort(key=lambda x: x[1], reverse=True)

            # Заполнили правую область чёрным
            screen.fill((0, 0, 0))
            # Нарисовали разделитель областей
            pg.draw.rect(screen, (255, 0, 0), (420, 0, 42, 420))


            sum_weights = 0
            for i in output_mas: sum_weights += i[1][0]
            output_mas = [[i[0], round(i[1][0] * 100 / sum_weights, 1)] for i in output_mas]
            # Прописать цифры
            for i in range(10):
                print_text(str(output_mas[i][0]), size + 10, i * 42)
                print_text(str(output_mas[i][1]) + '%', size + 60, i * 42, (0, 255, 0), 30)
                pg.draw.rect(screen, (255, 136, 0), (420 + 42, 21 + 42 * i, int(4 * output_mas[i][1]), 21))
            stage = 3

        clock.tick(fps)
        screen.blit(draw_screen, (0, 0))
        pg.display.flip()
    pg.quit()

    write_weights(n.wih, n.who)


# def train_number(epochs):
#     with open('real train.txt', 'r') as f:
#         for _ in range(epochs):
#             for line in f:
#                 input_list = np.array(list(map(float, line.split(' '))))
#                 right_num = int(input_list[0])
#                 answer = np.zeros(10) + 0.01
#                 print(right_num)
#                 answer[right_num] = 0.99
#                 n.train(input_list[1:], answer)
#     write_weights(n.wih, n.who)

what_number()

# print('1. Draw')
# print('2. Train')
# if input() == '1':
#     what_number()
# else:
#     print('epochs =', end=' ')
#     train_number(int(input()))





