import os
import logging
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from document_processor_langchain import process_document, search_documents, get_document_info

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
        [KeyboardButton("📋 Список инструментов"), KeyboardButton("📄 Загрузить документ")],
        [KeyboardButton("📊 Информация о документах")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        'Привет! Я бот для работы с документами.\n'
        'Я могу:\n'
        '• Загружать и обрабатывать документы (PDF, DOC, DOCX, TXT)\n'
        '• Искать информацию в загруженных документах\n'
        '• Показывать статистику по документам',
        reply_markup=reply_markup
    )

# Обработчики сообщений
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает текстовые сообщения."""
    text = update.message.text
    
    if text == "📋 Список инструментов":
        tools_list = """
📋 Доступные инструменты:
1. 📄 Загрузка документов (PDF, DOC, DOCX, TXT)
2. 🔍 Поиск по документам (просто отправьте текст)
3. 📊 Просмотр информации о загруженных документах
        """
        await update.message.reply_text(tools_list)
    elif text == "📄 Загрузить документ":
        await update.message.reply_text(
            "Пожалуйста, отправьте документ, который хотите загрузить.\n"
            "Поддерживаемые форматы: PDF, DOC, DOCX, TXT"
        )
    elif text == "📊 Информация о документах":
        info = get_document_info()
        if info["total_documents"] > 0:
            response = f"📊 Всего документов: {info['total_documents']}\n\n"
            for doc in info["documents"]:
                response += f"📄 {os.path.basename(doc['source'])}\n"
                response += f"   Чанков: {doc['chunks']}\n\n"
        else:
            response = "📭 В базе пока нет документов"
        await update.message.reply_text(response)
    else:
        # Поиск по документам
        results = search_documents(text)
        if results:
            response = "🔍 Результаты поиска:\n\n"
            for i, result in enumerate(results, 1):
                response += f"{i}. {result['text'][:200]}...\n"
                response += f"   Источник: {os.path.basename(result['metadata']['source'])}\n"
                response += f"   Релевантность: {result['score']:.2f}\n\n"
            await update.message.reply_text(response)
        else:
            await update.message.reply_text(
                "🔍 По вашему запросу ничего не найдено.\n"
                "Попробуйте изменить формулировку или загрузите новые документы."
            )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает сообщения с документами."""
    document = update.message.document
    file_name = document.file_name
    
    # Проверка расширения файла
    allowed_extensions = ['.pdf', '.doc', '.docx', '.txt']
    file_extension = os.path.splitext(file_name)[1].lower()
    
    if file_extension not in allowed_extensions:
        await update.message.reply_text(
            f"❌ Неподдерживаемый формат файла: {file_extension}\n"
            f"Поддерживаемые форматы: {', '.join(allowed_extensions)}"
        )
        return

    # Скачивание файла
    file = await context.bot.get_file(document.file_id)
    file_path = f"temp_{file_name}"
    await file.download_to_drive(file_path)
    
    # Обработка документа
    success = process_document(file_path)
    
    # Удаление временного файла
    try:
        os.remove(file_path)
    except:
        pass
    
    if success:
        await update.message.reply_text(
            f"✅ Документ успешно обработан и сохранен!\n"
            f"📄 Имя файла: {file_name}\n"
            f"📊 Тип файла: {document.mime_type}\n"
            f"📦 Размер файла: {document.file_size} байт"
        )
    else:
        await update.message.reply_text(
            "❌ Произошла ошибка при обработке документа.\n"
            "Пожалуйста, проверьте формат файла и попробуйте снова."
        )

def main():
    """Запускает бота."""
    # Создаем приложение
    application = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()

    # Добавляем обработчики команд
    application.add_handler(CommandHandler("start", start))

    # Добавляем обработчики сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Запускаем бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 