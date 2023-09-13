import torch
from libnway.nway_coverage import nway_align

src = "The cat sits on the mat"
# tms = [
#     "The dog sits on the ground",
#     "The cat sleeps on the bed ."
# ]
tms = [
    "The cat sits on it",
    "it's on the mat where the cat sits on"
]

src = src.split(" ")
tms = [tm.split(" ") for tm in tms]

nway_align(src, tms)

