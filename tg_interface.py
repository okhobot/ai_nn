import telebot
import nn



TOKEN = ""
with open("config/tg_token.txt") as f:
    TOKEN=f.read()

# Создаем экземпляр бота
bot = telebot.TeleBot(TOKEN)

neuro=nn.NN("tiiuae/Falcon-H1-1.5B-Instruct-GGUF", "Falcon-H1-1.5B-Instruct-Q4_0.gguf","config/hf_token.txt",True, 8192,6,2)
with open("config/tg_init_prompt.txt", encoding="utf-8") as f:
    print(neuro.chat(f.read()))


def call_typing_event(message):
    if message.chat.type == 'private':
        bot.send_chat_action(message.chat.id, 'typing')
    # В группе - с thread_id
    elif message.chat.type in ['group', 'supergroup']:
        bot.send_chat_action(
            message.chat.id, 
            'typing',
            message_thread_id=message.message_thread_id if hasattr(message, 'message_thread_id') else None
        )

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, ".")

# Обработчик команды /help
@bot.message_handler(commands=['help'])
def send_help(message):
    help_text = """
Доступные команды:
/help - показать справку
Для общения ответьте на сообщение бота.
    """
    bot.reply_to(message, help_text)

# Обработчик команды /about
@bot.message_handler(func=lambda message: message.reply_to_message is not None and  message.reply_to_message.from_user.id == bot.get_me().id)
def send_about(message):
    call_typing_event(message)
    inp=message.from_user.username+": "+message.text
    print(inp)
    bot.reply_to(message, neuro.chat(inp,128))


# Запуск бота
if __name__ == "__main__":
    print("Бот запущен...")
    bot.polling(none_stop=True)