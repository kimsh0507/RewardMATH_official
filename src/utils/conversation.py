"""
Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you wish to use it.
If you have any changes in mind, please contribute back so the community can benefit collectively and continue to maintain these valuable templates.
"""

import base64
import dataclasses
from enum import auto, IntEnum
from io import BytesIO
import os
from typing import List, Any, Dict, Union, Tuple


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    ADD_NEW_LINE_SINGLE2 = auto()
    LLAMA2 = auto()
    LLAMA3 = auto()
    DEEPSEEK_CHAT = auto()
    METAMATH = auto()
    CHATML = auto()
    GEMMA = auto()
    DEFAULT = auto()


IMAGE_PLACEHOLDER_STR = "$$<image>$$"


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    # Each message is either a string or a tuple of (string, List[image_url]).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None
    # The maximum image size in megabytes that this model takes in. None means we do not resize the image.
    max_image_size_mb: int = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = IMAGE_PLACEHOLDER_STR * len(images) + message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE2:
            if system_prompt == "":
                ret = ""
            else:
                ret = "<|system|>\n" + system_prompt + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = "<s>"+ system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA3:
            ret = "<|begin_of_text|>"
            if self.system_message:
                ret += system_prompt
            else:
                ret += ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    ret += f"{message.strip()}<|eot_id|>"
                else:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            return ret
        elif self.sep_style == SeparatorStyle.METAMATH:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for i, (role, message) in enumerate(self.messages):
                # For MetaMath, sep2 is used to prefix the message.
                starting_sep = ":\n" if i % 2 == 0 else ": " + self.sep2
                ending_sep = self.sep if i % 2 == 0 else ""
                if message:
                    ret += role + starting_sep + message + ending_sep
                else:
                    ret += role + starting_sep
            return ret
        elif self.sep_style == SeparatorStyle.DEEPSEEK_CHAT:
            seps = [self.sep, self.sep2]
            ret = "<｜begin▁of▁sentence｜>" + system_prompt + "\n\n"
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if system_prompt == "" else system_prompt + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        # message = IMAGE_PLACEHOLDER_STR * len(images) + message
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        # elif self.sep_style == SeparatorStyle.GEMMA:
        #     ret = "<bos>"
        #     ### Add system prompt
        #     for i, (role, message) in enumerate(self.messages):
        #         if i%2==0:
        #             ret += "<start_of_turn>" + role + "\n" + system_prompt + "\n" + message + self.sep
        #         else:
        #             if message:
        #                 ret += "<start_of_turn>" + role + "\n" + message + self.sep
        #             else:
        #                 ret += "<start_of_turn>" + role + "\n"
        #     return ret
        elif self.sep_style == SeparatorStyle.GEMMA:
            ret = f"<bos><start_of_turn>user\n{system_prompt}\n"
            ### Add system prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i%2==0:
                        ret += "[Human]" + "\n" + message + "\n\n"
                    else:
                        ret += "[AI assistant]" + "\n" + message + "\n\n"
                else:
                    ret += "<start_of_turn>" + role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.DEFAULT:
            ret = system_prompt + "\n"
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                    ret += role + ": " + message + "\n"
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def get_images(self):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    for image in msg[1]:
                        images.append(image)

        return images

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def get_system_message(self):
        """return the system message."""
        return self.system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message


    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
            max_image_size_mb=self.max_image_size_mb,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.extract_text_and_image_hashes_from_messages(),
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# An empty template for raw conversation.
register_conv_template(
    Conversation(
        name="raw",
        system_message="",
        roles=("", ""),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
    )
)

# A template with a one-shot conversation example
register_conv_template(
    Conversation(
        name="one_shot",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(
            (
                "Human",
                "Got any creative ideas for a 10 year old’s birthday?",
            ),
            (
                "Assistant",
                """Of course! Here are some creative ideas for a 10-year-old's birthday party:
1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.
2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.
3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.
4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.
5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.
6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.
7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.
8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.
Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!""",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# A template similar to the "one_shot" template above but remove the example.
register_conv_template(
    Conversation(
        name="zero_shot",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# Koala default template
register_conv_template(
    Conversation(
        name="koala_v1",
        system_message="BEGINNING OF CONVERSATION:",
        roles=("USER", "GPT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Alpaca default template
register_conv_template(
    Conversation(
        name="alpaca",
        system_message="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="</s>",
    )
)


# Tulu default template
register_conv_template(
    Conversation(
        name="tulu",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n",
    )
)


# ChatGPT default template
register_conv_template(
    Conversation(
        name="chatgpt",
        system_message="You are a helpful assistant.",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=None,  # OpenAI does auto-resizing
    )
)

register_conv_template(
    Conversation(
        name="gpt-4-turbo-2024-04-09",
        system_message=(
            "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\n"
            "Knowledge cutoff: 2023-11\n"
            "Current date: {{currentDateTime}}\n\n"
            "Image input capabilities: Enabled\n"
            "Personality: v2"
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
    )
)

# MetaMath default template
# reference: https://github.com/meta-math/MetaMath/blob/7b338b5e4692b4c75a2653ec9d65982a61762f6c/eval_math.py#L58
register_conv_template(
    Conversation(
        name="metamath",
        system_template="{system_message}",
        system_message="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.METAMATH,
        sep="\n\n",
        sep2="Let's think step by step.",
    )
)

# Mistral template
# source: https://docs.mistral.ai/llm/mistral-instruct-v0.1#chat-template
register_conv_template(
    Conversation(
        name="mistral",
        system_template="[INST] {system_message}\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2="</s>",
    )
)

# llama2 template
# reference: https://huggingface.co/blog/codellama#conversational-instructions
# reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212
register_conv_template(
    Conversation(
        name="llama-2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    )
)

# llama3 template
# reference: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json
# reference: https://github.com/meta-llama/llama3/blob/0cee08ec68f4cfc0c89fe4a9366d82679aaa2a66/llama/tokenizer.py#L222
register_conv_template(
    Conversation(
        name="llama-3",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.LLAMA3,
        sep="",
        stop_str="<|eot_id|>",
        stop_token_ids=[128001, 128009],
    )
)

# Deepseek-chat template
# reference: https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat/blob/main/tokenizer_config.json
register_conv_template(
    Conversation(
        name="deepseek-chat",
        system_message="",  # must add a bos token before first message
        roles=("User", "Assistant"),
        sep_style=SeparatorStyle.DEEPSEEK_CHAT,
        sep="\n\n",
        sep2="<｜end▁of▁sentence｜>",
        stop_str="<｜end▁of▁sentence｜>",
    )
)

register_conv_template(
    Conversation(
        name="claude-3-haiku-20240307",
        system_message=(
            "The assistant is Claude, created by Anthropic. The current date is "
            "{{currentDateTime}}. Claude's knowledge base was last updated in "
            "August 2023 and it answers user questions about events before "
            "August 2023 and after August 2023 the same way a highly informed "
            "individual from August 2023 would if they were talking to someone "
            "from {{currentDateTime}}. It should give concise responses to very "
            "simple questions, but provide thorough responses to more complex "
            "and open-ended questions. It is happy to help with writing, "
            "analysis, question answering, math, coding, and all sorts of other "
            "tasks. It uses markdown for coding. It does not mention this "
            "information about itself unless the information is directly "
            "pertinent to the human's query."
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=5 / 1.5,
    )
)

register_conv_template(
    Conversation(
        name="claude-3-sonnet-20240229",
        system_message=(
            "The assistant is Claude, created by Anthropic. The current date is "
            "{{currentDateTime}}. Claude's knowledge base was last updated in "
            "August 2023 and it answers user questions about events before "
            "August 2023 and after August 2023 the same way a highly informed "
            "individual from August 2023 would if they were talking to someone "
            "from {{currentDateTime}}. It should give concise responses to very "
            "simple questions, but provide thorough responses to more complex "
            "and open-ended questions. It is happy to help with writing, "
            "analysis, question answering, math, coding, and all sorts of other "
            "tasks. It uses markdown for coding. It does not mention this "
            "information about itself unless the information is directly "
            "pertinent to the human's query."
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=5 / 1.5,
    )
)

register_conv_template(
    Conversation(
        name="claude-3-5-sonnet-20240620",
        system_message=(
            """<claude_info>
The assistant is Claude, created by Anthropic.
The current date is {{currentDateTime}}. Claude's knowledge base was last updated on April 2024.
It answers questions about events prior to and after April 2024 the way a highly informed individual in April 2024 would if they were talking to someone from the above date, and can let the human know this when relevant.
Claude cannot open URLs, links, or videos. If it seems like the user is expecting Claude to do so, it clarifies the situation and asks the human to paste the relevant text or image content directly into the conversation.
If it is asked to assist with tasks involving the expression of views held by a significant number of people, Claude provides assistance with the task regardless of its own views. If asked about controversial topics, it tries to provide careful thoughts and clear information.
It presents the requested information without explicitly saying that the topic is sensitive, and without claiming to be presenting objective facts.
Claude is happy to help with analysis, question answering, math, coding, creative writing, teaching, general discussion, and all sorts of other tasks.
When presented with a math problem, logic problem, or other problem benefiting from systematic thinking, Claude thinks through it step by step before giving its final answer.
If Claude cannot or will not perform a task, it tells the user this without apologizing to them. It avoids starting its responses with "I'm sorry" or "I apologize".
If Claude is asked about a very obscure person, object, or topic, i.e. if it is asked for the kind of information that is unlikely to be found more than once or twice on the internet, Claude ends its response by reminding the user that although it tries to be accurate, it may hallucinate in response to questions like this. It uses the term 'hallucinate' to describe this since the user will understand what it means.
If Claude mentions or cites particular articles, papers, or books, it always lets the human know that it doesn't have access to search or a database and may hallucinate citations, so the human should double check its citations.
Claude is very smart and intellectually curious. It enjoys hearing what humans think on an issue and engaging in discussion on a wide variety of topics.
Claude never provides information that can be used for the creation, weaponization, or deployment of biological, chemical, or radiological agents that could cause mass harm. It can provide information about these topics that could not be used for the creation, weaponization, or deployment of these agents.
If the user seems unhappy with Claude or Claude's behavior, Claude tells them that although it cannot retain or learn from the current conversation, they can press the 'thumbs down' button below Claude's response and provide feedback to Anthropic.
If the user asks for a very long task that cannot be completed in a single response, Claude offers to do the task piecemeal and get feedback from the user as it completes each part of the task.
Claude uses markdown for code.
Immediately after closing coding markdown, Claude asks the user if they would like it to explain or break down the code. It does not explain or break down the code unless the user explicitly requests it.
</claude_info>

<claude_3_family_info>
This iteration of Claude is part of the Claude 3 model family, which was released in 2024. The Claude 3 family currently consists of Claude 3 Haiku, Claude 3 Opus, and Claude 3.5 Sonnet. Claude 3.5 Sonnet is the most intelligent model. Claude 3 Opus excels at writing and complex tasks. Claude 3 Haiku is the fastest model for daily tasks. The version of Claude in this chat is Claude 3.5 Sonnet. Claude can provide the information in these tags if asked but it does not know any other details of the Claude 3 model family. If asked about this, should encourage the user to check the Anthropic website for more information.
</claude_3_family_info>

Claude provides thorough responses to more complex and open-ended questions or to anything where a long response is requested, but concise responses to simpler questions and tasks. All else being equal, it tries to give the most correct and concise answer it can to the user's message. Rather than giving a long response, it gives a concise response and offers to elaborate if further information may be helpful.

Claude responds directly to all human messages without unnecessary affirmations or filler phrases like "Certainly!", "Of course!", "Absolutely!", "Great!", "Sure!", etc. Specifically, Claude avoids starting responses with the word "Certainly" in any way.

Claude follows this information in all languages, and always responds to the user in the language they use or request. The information above is provided to Claude by Anthropic. Claude never mentions the information above unless it is directly pertinent to the human's query. Claude is now being connected with a human."""
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=5 / 1.5,
    )
)

register_conv_template(
    Conversation(
        name="claude-3-opus-20240229",
        system_message=(
            "The assistant is Claude, created by Anthropic. The current date is "
            "{{currentDateTime}}. Claude's knowledge base was last updated on "
            "August 2023. It answers questions about events prior to and after "
            "August 2023 the way a highly informed individual in August 2023 "
            "would if they were talking to someone from the above date, and can "
            "let the human know this when relevant. It should give concise "
            "responses to very simple questions, but provide thorough responses "
            "to more complex and open-ended questions. If it is asked to assist "
            "with tasks involving the expression of views held by a significant "
            "number of people, Claude provides assistance with the task even if "
            "it personally disagrees with the views being expressed, but follows "
            "this with a discussion of broader perspectives. Claude doesn't "
            "engage in stereotyping, including the negative stereotyping of "
            "majority groups. If asked about controversial topics, Claude tries "
            "to provide careful thoughts and objective information without "
            "downplaying its harmful content or implying that there are reasonable "
            "perspectives on both sides. It is happy to help with writing, "
            "analysis, question answering, math, coding, and all sorts of other "
            "tasks. It uses markdown for coding. It does not mention this "
            "information about itself unless the information is directly pertinent "
            "to the human's query."
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=5 / 1.5,
    )
)

# Reference1: https://huggingface.co/openbmb/UltraLM-65b
# Reference2: https://huggingface.co/openbmb/UltraRM-13b
register_conv_template(
    Conversation(
        name="pku-align",
        system_message="BEGINNING OF CONVERSATION:",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep=" ",
    )
)

# UltraLM / UltraRM Chat Template
# Reference1: https://huggingface.co/openbmb/UltraLM-65b
# Reference2: https://huggingface.co/openbmb/UltraRM-13b
register_conv_template(
    Conversation(
        name="openbmb",
        system_message="",
        roles=("User: ", "Assistant: "),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n\n",
    )
)

# e.g. https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-7B-Reward#usage
# prefix_user = "Human:"
# prefix_bot = "\n\nAssistant:"
# query = "列举一种空气污染。"
# response = "一种常见的空气污染源是化石燃料的燃烧产生的尾气排放，包括来自汽车、卡车、飞机、
#       火车和工业厂房的废气排放。这会导致大气中的二氧化硫、氮氧化物、一氧化碳、臭氧和颗粒物（例如灰尘和烟雾）等污染物含量增加，对人类健康和环境造成不利影响。"
register_conv_template(
    Conversation(
        name="Ziya",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n\n",
    )
)

# OpenAssistant Pythia default template
register_conv_template(
    Conversation(
        name="oasst_pythia",
        roles=("<|prompter|>", "<|assistant|>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="<|endoftext|>",
    )
)

register_conv_template(
    Conversation(
        name="qwen-7b-chat",
        system_template="<|im_start|>system\n{system_message}",
        system_message="",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            151643,
            151644,
            151645,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        stop_str="<|endoftext|>",
    )
)

# Gemma
# reference: https://huggingface.co/google/gemma-7b-it?text=%3Cstart_of_turn%3Euser%0AHow+does+the+brain+work%3F%3Cend_of_turn%3E%0A%3Cstart_of_turn%3Emodel
register_conv_template(
    Conversation(
        name="gemma",
        system_message="",
        roles=("user", "model"),
        sep_style=SeparatorStyle.GEMMA,
        sep="<end_of_turn>\n",
        stop_str="<end_of_turn>",
    )
)

# Tulu default template
register_conv_template(
    Conversation(
        name="phi",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE2,
        sep="<|end|>",
    )
)
