import os
import logging
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Обработчики команд
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет сообщение при выполнении команды /start."""
    # Создаем клавиатуру с кнопками
    keyboard = [
        [KeyboardButton("📋 Список инструментов"), KeyboardButton("📄 Загрузить документ")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        'Привет! Я бот, который может обрабатывать различные типы сообщений.\n'
        'Отправьте мне текст, фото, документ или стикер, и я отвечу!',
        reply_markup=reply_markup
    )

# Обработчики сообщений
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает текстовые сообщения."""
    text = update.message.text
    await update.message.reply_text(f'Вы отправили текст: {text}')

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает сообщения с фотографиями."""
    photo = update.message.photo[-1]  # Получаем фото наилучшего качества
    await update.message.reply_text(
        f'Вы отправили фото!\n'
        f'Размер файла: {photo.file_size} байт\n'
        f'Разрешение: {photo.width}x{photo.height}'
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает сообщения с документами."""
    document = update.message.document
    await update.message.reply_text(
        f'Вы отправили документ!\n'
        f'Имя файла: {document.file_name}\n'
        f'Тип файла: {document.mime_type}\n'
        f'Размер файла: {document.file_size} байт'
    )

async def handle_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает сообщения со стикерами."""
    sticker = update.message.sticker
    await update.message.reply_text(
        f'Вы отправили стикер!\n'
        f'ID стикера: {sticker.file_id}\n'
        f'Эмодзи: {sticker.emoji}'
    )

async def handle_tools_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает нажатие кнопки 'Список инструментов'."""
    tools_list = """
📋 Доступные инструменты:
1. Обработка текста
2. Обработка фото
3. Обработка документов
4. Обработка стикеров
    """
    await update.message.reply_text(tools_list)

async def handle_load_doc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает нажатие кнопки 'Загрузить документ'."""
    await update.message.reply_text(
        "Пожалуйста, отправьте документ, который хотите загрузить.\n"
        "Поддерживаемые форматы: PDF, DOC, DOCX, TXT"
    )

def main():
    """Запускает бота."""
    # Создаем приложение
    application = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()

    # Добавляем обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Добавляем обработчики сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.STICKER, handle_sticker))

    # Добавляем обработчики кнопок
    application.add_handler(MessageHandler(filters.Regex("^📋 Список инструментов$"), handle_tools_list))
    application.add_handler(MessageHandler(filters.Regex("^📄 Загрузить документ$"), handle_load_doc))

    # Запускаем бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 