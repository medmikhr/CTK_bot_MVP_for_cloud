import os
import logging
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from dotenv import load_dotenv
from document_processor_langchain import process_document, get_document_info
from agent import agent_ask

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Получение токена из переменных окружения
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not TOKEN:
    raise ValueError("Не найден токен бота в переменных окружения")

# Словарь коллекций и их пользователей
COLLECTIONS = {
    "ctk": [
        673473862,  # Замените на реальные ID пользователей ЦТК
    ],
    "sbf": [
        135727236,  # Замените на реальные ID пользователей СБФ
    ]
}

# Словарь для хранения состояний пользователей
user_states = {}

def get_user_collection(user_id: int) -> str:
    """Определяет, к какой коллекции имеет доступ пользователь"""
    for collection, users in COLLECTIONS.items():
        if user_id in users:
            return collection
    return None

def get_main_keyboard(user_id: int = None):
    """Создает основную клавиатуру"""
    keyboard = [
        [KeyboardButton("/tools_list")],
        [KeyboardButton("/docs_list")]
    ]
    
    # Добавляем кнопку загрузки только для пользователей с правами
    if get_user_collection(user_id):
        keyboard.insert(1, [KeyboardButton("/load_doc")])
    
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    help_text = (
        "Привет! я Data Governance бот для повышения культуры работы с данными."
    )
    await update.message.reply_text(help_text, reply_markup=get_main_keyboard())

async def tools_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /tools_list"""
    user_id = update.effective_user.id
    response = agent_ask(user_id, "какие инструменты тебе доступны?")
    await update.message.reply_text(response.content)

async def load_doc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /load_doc"""
    user_id = update.effective_user.id
    collection = get_user_collection(user_id)
    
    # Проверяем, есть ли у пользователя доступ к какой-либо коллекции
    if not collection:
        await update.message.reply_text(
            "⛔ У вас нет прав для загрузки документов. "
            "Обратитесь к администратору для получения доступа."
        )
        return
    
    user_states[user_id] = 'waiting_for_document'
    message_text = (
        f"📤 Пожалуйста, отправьте документ для загрузки в коллекцию {collection.upper()}.\n"
        "Поддерживаемые форматы: PDF, DOC, DOCX, TXT"
    )
    await update.message.reply_text(message_text)

async def docs_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /docs_list"""
    info = get_document_info()
    if info["total_documents"] == 0:
        message_text = "📊 В базе данных пока нет документов."
    else:
        response = f"📊 Всего документов: {info['total_documents']}\n\n"
        for doc in info["documents"]:
            response += f"📄 {os.path.basename(doc['source'])}\n"
            response += f"   Чанков: {doc['chunks']}\n"
        message_text = response
    await update.message.reply_text(message_text)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик получения документов"""
    user_id = update.effective_user.id
    collection = get_user_collection(user_id)
    
    # Проверяем, есть ли у пользователя доступ к какой-либо коллекции
    if not collection:
        await update.message.reply_text(
            "⛔ У вас нет прав для загрузки документов. "
            "Обратитесь к администратору для получения доступа."
        )
        return
    
    # Проверяем, ожидаем ли мы документ от этого пользователя
    if user_id not in user_states or user_states[user_id] != 'waiting_for_document':
        await update.message.reply_text(
            "❌ Пожалуйста, сначала нажмите кнопку '/load_doc' или используйте команду /load_doc"
        )
        return
    
    try:
        # Получаем информацию о файле
        file = await context.bot.get_file(update.message.document.file_id)
        file_name = update.message.document.file_name
        
        # Проверка расширения файла
        allowed_extensions = ['.pdf', '.doc', '.docx', '.txt']
        file_ext = os.path.splitext(file_name)[1].lower()
        
        if file_ext not in allowed_extensions:
            await update.message.reply_text(
                f"❌ Неподдерживаемый формат файла: {file_ext}\n"
                f"Поддерживаемые форматы: {', '.join(allowed_extensions)}"
            )
            return
        
        # Скачиваем файл
        file_path = f"temp_{file_name}"
        await file.download_to_drive(file_path)
        
        # Обрабатываем документ
        if process_document(file_path, collection=collection):
            await update.message.reply_text(f"✅ Документ успешно обработан и добавлен в коллекцию {collection.upper()}: {file_name}")
        else:
            await update.message.reply_text("❌ Ошибка при обработке документа")
        
        # Удаляем временный файл
        os.remove(file_path)
        
        # Сбрасываем состояние пользователя
        user_states.pop(user_id, None)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке документа: {e}")
        await update.message.reply_text("❌ Произошла ошибка при обработке документа")
        # Сбрасываем состояние пользователя в случае ошибки
        user_states.pop(user_id, None)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    try:
        user_id = update.effective_user.id
        text = update.message.text
        
        # Отправляем сообщение о том, что запрос обрабатывается
        processing_message = await update.message.reply_text("🤔 Обрабатываю ваш запрос...")
        
        # Передаем запрос агенту
        response = agent_ask(user_id, text)
        
        # Удаляем сообщение о обработке
        await processing_message.delete()
        
        # Отправляем ответ
        await update.message.reply_text(response.content)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке текстового сообщения: {e}")
        await update.message.reply_text(
            "❌ Произошла ошибка при обработке вашего запроса. "
            "Пожалуйста, попробуйте еще раз или обратитесь к администратору."
        )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на кнопки"""
    query = update.callback_query
    
    # Вызываем соответствующую функцию в зависимости от нажатой кнопки
    if query.data == "tools_list":
        await tools_list(update, context)
    elif query.data == "load_doc":
        await load_doc(update, context)
    elif query.data == "docs_list":
        await docs_list(update, context)
    
    # Отвечаем на callback query
    await query.answer()

async def handle_other_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик для всех остальных типов сообщений"""
    await update.message.reply_text(
        "❌ Я могу обрабатывать только текстовые сообщения и документы.\n"
        "Пожалуйста, отправьте текстовый запрос или документ."
    )

def main():
    """Основная функция запуска бота"""
    # Создаем приложение
    application = Application.builder().token(TOKEN).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("tools_list", tools_list))
    application.add_handler(CommandHandler("load_doc", load_doc))
    application.add_handler(CommandHandler("docs_list", docs_list))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.ALL, handle_other_messages))  # Обработчик для всех остальных типов сообщений
    application.add_handler(CallbackQueryHandler(button_callback))  # Обработчик кнопок
    
    # Запускаем бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 