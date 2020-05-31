from aiogram import Bot, Dispatcher
from aiogram.types import base
import datetime
from aiogram import types
import typing


class ShopBot(Bot):
    async def send_photo(self, user_input, tonality, photo_url, chat_id: typing.Union[base.Integer, base.String],
                         photo: typing.Union[base.InputFile, base.String],
                         caption: typing.Union[base.String, None] = None,
                         parse_mode: typing.Union[base.String, None] = None,
                         disable_notification: typing.Union[base.Boolean, None] = None,
                         reply_to_message_id: typing.Union[base.Integer, None] = None,
                         reply_markup: typing.Union[types.InlineKeyboardMarkup,
                                                    types.ReplyKeyboardMarkup,
                                                    types.ReplyKeyboardRemove,
                                                    types.ForceReply, None] = None) -> types.Message:
        ShopBot._write_log(chat_id, user_input, tonality, "{}: {}".format(caption, photo_url))
        return await super().send_photo(chat_id, photo, caption, parse_mode, disable_notification, reply_to_message_id,
                                        reply_markup)

    async def send_message(self, user_input, tonality, chat_id: typing.Union[base.Integer, base.String],
                           text: base.String,
                           parse_mode: typing.Union[base.String, None] = None,
                           disable_web_page_preview: typing.Union[base.Boolean, None] = None,
                           disable_notification: typing.Union[base.Boolean, None] = None,
                           reply_to_message_id: typing.Union[base.Integer, None] = None,
                           reply_markup: typing.Union[types.InlineKeyboardMarkup,
                                                      types.ReplyKeyboardMarkup,
                                                      types.ReplyKeyboardRemove,
                                                      types.ForceReply, None] = None) -> types.Message:
        ShopBot._write_log(chat_id, user_input, tonality, text)
        return await super().send_message(chat_id, text, parse_mode, disable_web_page_preview, disable_notification,
                                          reply_to_message_id, reply_markup)

    @staticmethod
    def _write_log(user_id, text, tonality, response):
        with open("logs/{}.txt".format(user_id), 'a', encoding='utf-8') as log_file:
            log_record = "[{}] | [{}]: {}->{}. Tonality: {}/{}\n".format(datetime.datetime.now(), user_id, text,
                                                                         response, tonality[0], tonality[1])
            log_file.write(log_record)
            print(log_record, end="")
