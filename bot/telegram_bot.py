from __future__ import annotations

import logging

from telegram import BotCommand, BotCommandScopeAllGroupChats, Update, constants
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, Application, ContextTypes

from ai_helper import AIHelper, localized_text
from utils import (
    is_group_chat, get_thread_id, message_text, wrap_with_indicator, split_into_chunks,
    is_allowed, is_within_budget, get_reply_to_message_id, add_chat_request_to_usage_tracker,
    error_handler, is_direct_result, handle_direct_result
)


class ChatGPTTelegramBot:
    """
    Class representing a ChatGPT Telegram Bot.
    """

    def __init__(self, config: dict, openai: AIHelper):
        """
        Initializes the bot with the given configuration and AI helper object.
        :param config: A dictionary containing the bot configuration
        :param openai: AIHelper object
        """
        self.config = config
        self.openai = openai
        bot_language = self.config['bot_language']

        # Define bot commands
        self.commands = [
            BotCommand(command='reset', description=localized_text('reset_description', bot_language))
        ]
        self.group_commands = [
                                  BotCommand(command='chat',
                                             description=localized_text('chat_description', bot_language))
                              ] + self.commands

        # Set up messages for disallowed users and budget limits
        self.disallowed_message = localized_text('disallowed', bot_language)
        self.budget_limit_message = localized_text('budget_limit', bot_language)

        # Initialize usage tracking
        self.usage = {}
        self.last_message = {}

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resets the conversation for the user.
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            'is not allowed to reset the conversation')
            await self.send_disallowed_message(update, context)
            return

        logging.info(f'Resetting the conversation for user {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})...')

        chat_id = update.effective_chat.id
        self.openai.reset_conversation(chat_id=chat_id)
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=localized_text('reset_done', self.config['bot_language'])
        )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handles the /start command.
        """
        prompt = 'Привет'  # TODO: Translate this
        await self.__prompt(prompt=prompt, chat_id=update.effective_chat.id, user_id=update.message.from_user.id,
                            update=update, context=context)

    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handles incoming messages and generates responses.
        """
        if update.edited_message or not update.message or update.message.via_bot:
            return

        if not await self.check_allowed_and_within_budget(update, context):
            return

        logging.info(
            f'New message received from user {update.message.from_user.name} (id: {update.message.from_user.id})')

        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        prompt = message_text(update.message)
        self.last_message[chat_id] = prompt

        await self.__prompt(prompt=prompt, chat_id=chat_id, user_id=user_id, update=update, context=context)

    async def __prompt(self, prompt: str, chat_id: int, user_id: int, update: Update,
                       context: ContextTypes.DEFAULT_TYPE):
        """
        Internal method to process the prompt and generate a response.
        """
        if is_group_chat(update):
            trigger_keyword = self.config['group_trigger_keyword']

            if prompt.lower().startswith(trigger_keyword.lower()) or update.message.text.lower().startswith('/chat'):
                if prompt.lower().startswith(trigger_keyword.lower()):
                    prompt = prompt[len(trigger_keyword):].strip()

                if update.message.reply_to_message and \
                        update.message.reply_to_message.text and \
                        update.message.reply_to_message.from_user.id != context.bot.id:
                    prompt = f'"{update.message.reply_to_message.text}" {prompt}'
            else:
                if update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id:
                    logging.info('Message is a reply to the bot, allowing...')
                else:
                    logging.warning('Message does not start with trigger keyword, ignoring...')
                    return

        try:
            total_tokens = 0

            async def _reply():
                nonlocal total_tokens
                response, total_tokens = await self.openai.get_chat_response(chat_id=chat_id, query=prompt)

                if is_direct_result(response):
                    return await handle_direct_result(self.config, update, response)

                chunks = split_into_chunks(response)

                for index, chunk in enumerate(chunks):
                    try:
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(self.config, update) if index == 0 else None,
                            text=chunk,
                            parse_mode=constants.ParseMode.MARKDOWN
                        )
                    except Exception:
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(self.config, update) if index == 0 else None,
                            text=chunk
                        )

            await wrap_with_indicator(update, context, _reply, constants.ChatAction.TYPING)

            add_chat_request_to_usage_tracker(self.usage, self.config, user_id, total_tokens)

        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f"{localized_text('chat_fail', self.config['bot_language'])} {str(e)}",
                parse_mode=constants.ParseMode.MARKDOWN
            )

    async def check_allowed_and_within_budget(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """
        Checks if the user is allowed to use the bot and is within their budget.
        """
        name = update.message.from_user.name
        user_id = update.message.from_user.id

        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {name} (id: {user_id}) is not allowed to use the bot')
            await self.send_disallowed_message(update, context)
            return False
        if not is_within_budget(self.config, self.usage, update):
            logging.warning(f'User {name} (id: {user_id}) reached their usage limit')
            await self.send_budget_reached_message(update, context)
            return False

        return True

    async def send_disallowed_message(self, update: Update, _: ContextTypes.DEFAULT_TYPE):
        """
        Sends the disallowed message to the user.
        """
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=self.disallowed_message,
            disable_web_page_preview=True
        )

    async def send_budget_reached_message(self, update: Update, _: ContextTypes.DEFAULT_TYPE):
        """
        Sends the budget reached message to the user.
        """
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=self.budget_limit_message
        )

    async def post_init(self, application: Application) -> None:
        """
        Post initialization hook for the bot.
        """
        await application.bot.set_my_commands(self.group_commands, scope=BotCommandScopeAllGroupChats())
        await application.bot.set_my_commands(self.commands)

    def run(self):
        """
        Runs the bot indefinitely until the user presses Ctrl+C
        """
        application = ApplicationBuilder() \
            .token(self.config['token']) \
            .proxy_url(self.config['proxy']) \
            .get_updates_proxy_url(self.config['proxy']) \
            .post_init(self.post_init) \
            .concurrent_updates(True) \
            .build()

        application.add_handler(CommandHandler('reset', self.reset))
        application.add_handler(CommandHandler('start', self.start))
        application.add_handler(
            CommandHandler('chat', self.prompt, filters=filters.ChatType.GROUP | filters.ChatType.SUPERGROUP))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.prompt))

        application.add_error_handler(error_handler)

        application.run_polling()
