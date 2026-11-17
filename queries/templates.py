"""Prompt frames for Comma, a base model without a chat template.

A WildChat request is placed in a plain-text dialogue that the model
continues; generation stops when the model starts the next user turn. The
demonstrations set length and register of the answers, so they are part of
the benchmark definition: changing them changes the queries. They are
written for this purpose (not taken from WildChat or the Comma corpus), one
per task, and kept short so that most of the 4,096-token context is left for
the request and the answer.
"""

FEWSHOT = """\
The following is a conversation between a user and a knowledgeable, helpful assistant. \
The assistant answers questions accurately and writes creatively when asked.

User: Why is the sky blue?
Assistant: Sunlight contains all visible colours. When it passes through the atmosphere, \
it is scattered by the gas molecules in the air, and this scattering (Rayleigh scattering) \
is much stronger for short wavelengths than for long ones. Blue light is therefore scattered \
in all directions across the sky, while most of the red and yellow light passes straight \
through. That is why the sky looks blue during the day, and why sunsets look red: when the \
sun is low, its light travels through more air and most of the blue has been scattered away.

User: Write a short poem about a lighthouse keeper.
Assistant: He climbs the tower when the light goes grey,
and trims the wick the way he did last year;
the ships he guards will never know his name,
they only know the shore is somewhere near.

At dawn he logs the weather in his book,
the wind, the swell, the hour the lamp went dark,
then sleeps while all the harbour comes awake
and fishing boats head out beyond the mark.

User: {prompt}
Assistant:"""

PLAIN = """\
Question: {prompt}
Answer:"""

# name -> (frame, stop strings). Every frame ends right before the answer.
TEMPLATES = {
    "fewshot": (FEWSHOT, ["\nUser:"]),
    "plain": (PLAIN, ["\nQuestion:"]),
}


def render(name: str, prompt: str) -> str:
    return TEMPLATES[name][0].replace("{prompt}", prompt)


def stops(name: str) -> list[str]:
    return TEMPLATES[name][1]
