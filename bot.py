import typing

from aiogram import Bot, Dispatcher
from aiogram.types import base
import datetime
from aiogram import types
from aiogram.dispatcher import FSMContext
from aiogram import executor
import dialogflow
from bot.tgbot import ShopBot
from google.api_core.exceptions import InvalidArgument
from google.protobuf.json_format import MessageToDict
import os
import random
import re
import torch.nn
from ai.model import TextCNNDataset
import sys

from clean import process

session_static = "20203105_6"

USER_DATA_BASE = {}

TOKEN = 'TOKEN'


bot = ShopBot(token=TOKEN)

sys.path.insert(0, 'ai/')
model = torch.load('ai/TCNN-E750-L0.3312516510486603.pt')
dataset = TextCNNDataset(path='ai/data/train.second.json', lang="uk", vs=200000)


def get_prediction(text):
    """ Повертає сортований список (назва класу, ймовірність)"""
    text = process(text)
    embed_data = dataset.embed(text)
    intent_tensor = torch.tensor(embed_data)
    intent_tensor = TextCNNDataset.upsize(intent_tensor, 10).unsqueeze(0)

    output = model(intent_tensor)

    info = []
    for class_number, probability in enumerate(output[0]):
        info.append((dataset.get_label_by_index(class_number), probability))
    info.sort(key=lambda x: -x[1])
    return info


def diaflow(session_id, request):
    DIALOGFLOW_PROJECT_ID = 'myshop-rjpuxv'
    DIALOGFLOW_LANGUAGE_CODE = 'uk'

    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(DIALOGFLOW_PROJECT_ID, str(session_id) + session_static)
    text_input = dialogflow.types.TextInput(text=request, language_code=DIALOGFLOW_LANGUAGE_CODE)
    query_input = dialogflow.types.QueryInput(text=text_input)
    try:
        response = session_client.detect_intent(session=session, query_input=query_input)
    except InvalidArgument:
        raise

    query_result = MessageToDict(response.query_result)
    return query_result


def get_stuff(name, colors, size):
    all_files = []
    for color in colors:
        path = 'shop_data\\{name}\\{color}'.format(name=name, color=color)
        files = os.listdir(path)
        for file in files:
            all_files.append(os.path.join(path, file))
    return all_files


dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def start(message: types.Message, state: FSMContext):
    await bot.send_message(message.text, None, message.from_user.id,
                           "Вітаємо! Це чат бот яикй спробуе зрозуміти тебе по максимому. Почнімо.")


@dp.message_handler()
async def main(message: types.Message, state: FSMContext):
    # Local prediction for tonality
    local_prediction = get_prediction(message.text)

    # Get response from Dialogflow
    response = diaflow(message.chat.id, message.text)
    print(response)
    # Create DB record
    if not USER_DATA_BASE.get(message.from_user.id):
        USER_DATA_BASE[message.from_user.id] = {}

    if USER_DATA_BASE[message.from_user.id].get('waitPhone'):
        if re.match(r'(38)*(0*)(\d){9}', message.text):
            USER_DATA_BASE[message.from_user.id]['waitPhone'] = False
            # SAVE ORDER HERE
            return await bot.send_message(message.text, local_prediction, message.chat.id,
                                          "Дякуємо. Очікуйте дзвінка від менеджера")
        else:
            if response['intent']['displayName'] == "buyCancel":
                USER_DATA_BASE[message.from_user.id]['waitPhone'] = False
                await bot.send_message(message.text, local_prediction, message.chat.id, "Ок. Замовлення скасовано.")
            else:
                await bot.send_message(message.text, local_prediction, message.chat.id,
                                       "Для замовлення необхыдний ваш номер телефону, "
                                       "вкажыть його і з вами взяжеться менеджер")

    if response['intent']['displayName'] == "buyThis":
        if USER_DATA_BASE[message.from_user.id].get('last_stuff'):
            USER_DATA_BASE[message.from_user.id]['waitPhone'] = True
            return await bot.send_message(message.text, local_prediction, message.chat.id, "Гарний вибір, залиште "
                                                                                           "номер "
                                                                                           "телефону і наш менеджер з "
                                                                                           "вами зв'яжеться")
        else:
            response['action'] = 'input.unknown'

    if response.get('action') == 'input.unknown':
        positive, negative = local_prediction
        if positive > negative:
            return await bot.send_message(message.text, local_prediction, message.chat.id, "Ми раді що вам "
                                                                                           "сподобалось користуватись "
                                                                                           "нашим ботом. "
                                                                                           "Всього хорошого.")
        else:
            return await bot.send_message(message.text, local_prediction, message.chat.id,
                                          "Нам жаль що вам не сподобалось, Не засмучуйтесь, ваша розмова з ботом вплине"
                                          " на його майдутню поведінку і покращить її. Дякуємо!.")

    if response['intent']['displayName'] == "showMore":
        if USER_DATA_BASE.get(message.from_user.id):
            if USER_DATA_BASE[message.from_user.id].get('stuff'):
                USER_DATA_BASE[message.from_user.id]['last_stuff'] = random.choice(
                    USER_DATA_BASE[message.from_user.id].get('stuff'))
                photo = open(USER_DATA_BASE[message.from_user.id]['last_stuff'], 'rb')

                return await bot.send_photo(message.text, local_prediction, USER_DATA_BASE[message.from_user.id]['last_stuff'], message.chat.id, photo, caption="Як вам такий варіант?")

        await bot.send_message(message.text, local_prediction, message.chat.id, "Схоже ви ще не визначились що будете "
                                                                                "купувати. Спочатку виберіть тип "
                                                                                "товару.")

    if response.get('allRequiredParamsPresent'):
        if response['intent']['displayName'] == "buyTshirt":
            color = response['parameters']['color']
            size = response['parameters']['size']
            files = get_stuff('tshirt', color, size)
            USER_DATA_BASE[message.from_user.id]['intent'] = 'tshirt'
            USER_DATA_BASE[message.from_user.id]['stuff'] = files
            USER_DATA_BASE[message.from_user.id]['intent_set'] = True
            USER_DATA_BASE[message.from_user.id]['color'] = color
            USER_DATA_BASE[message.from_user.id]['size'] = size
            USER_DATA_BASE[message.from_user.id]['last_stuff'] = random.choice(files)
            photo = open(USER_DATA_BASE[message.from_user.id]['last_stuff'], 'rb')
            return await bot.send_photo(message.text, local_prediction, USER_DATA_BASE[message.from_user.id]['last_stuff'], message.chat.id, photo, caption="Як вам такий варіант?")

    await bot.send_message(message.text, local_prediction, message.chat.id, response['fulfillmentText'])


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
