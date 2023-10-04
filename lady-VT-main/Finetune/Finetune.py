import os
import openai

# export OPENAI_API_KEY="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"
openai --api-key "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" api fine_tunes.create -t InanisTest_prepared.jsonl -m davinci

openai --api-key "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" api fine_tunes.follow -i ft-51iNEgtEbPt1rEwhyXXEhrl1

!export OPENAI_API_KEY="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"; openai api fine_tunes.follow -i ft-51iNEgtEbPt1rEwhyXXEhrl1

openai --api-key "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" api fine_tunes.list