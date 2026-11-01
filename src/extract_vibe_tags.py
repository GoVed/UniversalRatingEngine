"""
extract_vibe_tags.py - Extract structured Narrative DNA & Vibe Tags for all titles.
"""

import polars as pl
import re

VIBE_RULES = {
    "Slow Burn": [
        r"\bslow burn\b", r"\bdeliberate pace\b", r"\bpatient\b", r"\bgradual\w*\b",
        r"\bmethodical\b", r"\bunhurried\b", r"\bslowly unravel\w*\b", r"\blanguid\b"
    ],
    "Mind-Bending Twist": [
        r"\btwist\b", r"\bplot twist\b", r"\bshocking revelation\b", r"\bmind-bending\b",
        r"\bastonishing turn\b", r"\bunexpected turn\b", r"\brevelation\b", r"\bdeception\b"
    ],
    "Time Loop / Paradox": [
        r"\btime loop\b", r"\btime travel\b", r"\btimeline\b", r"\btemporal\b",
        r"\balternate reality\b", r"\bparadox\b", r"\bpast and future\b", r"\btime warp\b"
    ],
    "Non-Linear Puzzle": [
        r"\bnon-linear\b", r"\bfragmented\b", r"\bjigsaw\b", r"\bflashbacks?\b",
        r"\bchronolog\w*\b", r"\binterwoven\b", r"\bmultiple perspectives\b"
    ],
    "Climax Payoff": [
        r"\bclimax\b", r"\bgrand finale\b", r"\beverything becomes clear\b", r"\bpieces fall into place\b",
        r"\bculminat\w+\b", r"\bfinal confrontation\b", r"\bpayoff\b", r"\bexplosive conclusion\b",
        r"\bfinal showdown\b", r"\bresolut\w+\b"
    ],
    "Unreliable Narrator": [
        r"\bunreliable\b", r"\bhallucination\b", r"\bdelusion\b", r"\bparanoia\b",
        r"\bsanity\b", r"\bdistort\w+\b", r"\bwhat is real\b", r"\bmadness\b", r"\bpsychosis\b"
    ],
    "Whodunit / Mystery": [
        r"\bdetective\b", r"\bmurder investigation\b", r"\bsuspect\b", r"\bcrime scene\b",
        r"\bclues?\b", r"\bwhodunit\b", r"\bhomicide\b", r"\bsolve the mystery\b"
    ],
    "Existential Dread": [
        r"\bexistential\b", r"\bcosmic\b", r"\bdread\b", r"\binsignificance\b",
        r"\bhopeless\b", r"\bdespair\b", r"\bbleak\b", r"\bvoid\b", r"\bmeaninglessness\b"
    ],
    "Psychological Warfare": [
        r"\bcat and mouse\b", r"\bmind games\b", r"\bintellectual duel\b", r"\bmanipulat\w+\b",
        r"\binterrogat\w+\b", r"\bstrategic\b", r"\bpsychological thriller\b", r"\bchess match\b"
    ],
    "Cerebral & Philosophical": [
        r"\bphilosophical\b", r"\bcerebral\b", r"\bethical dilemma\b", r"\bmoral dilemma\b",
        r"\bmetaphysic\w*\b", r"\bexistentialism\b", r"\bdeep questions\b", r"\bthe human condition\b"
    ],
    "Cozy & Wholesome": [
        r"\bwholesome\b", r"\bheartwarming\b", r"\bgentle\b", r"\bhealing\b",
        r"\bcomforting\b", r"\bcharming\b", r"\bcozy\b", r"\bsweet\b", r"\blighthearted\b", r"\bfriendship\b"
    ],
    "Atmospheric & Noir": [
        r"\batmospheric\b", r"\bnoir\b", r"\bneo-noir\b", r"\bhaunting\b",
        r"\bmoody\b", r"\bgothic\b", r"\bshadowy\b", r"\beerie\b", r"\brooding\b"
    ],
    "Melancholic & Bittersweet": [
        r"\bmelanchol\w*\b", r"\bbittersweet\b", r"\bpoignant\b", r"\bgrief\b",
        r"\bloss\b", r"\bheartbreaking\b", r"\bnostalgic\b", r"\bsadness\b", r"\bmourning\b"
    ],
    "High Adrenaline": [
        r"\badrenaline\b", r"\baction-packed\b", r"\bhigh-octane\b", r"\brelentless\b",
        r"\bexplosive\b", r"\bthrill ride\b", r"\bchase\b", r"\bfast-paced\b", r"\bvisceral\b"
    ],
    "Found Family": [
        r"\bfound family\b", r"\bunlikely allies\b", r"\bband of misfits\b", r"\bragtag\b",
        r"\bcompanionship\b", r"\bcamaraderie\b", r"\bcrew\b", r"\bbrotherhood\b"
    ],
    "Moral Ambiguity": [
        r"\bmoral ambiguity\b", r"\bgrey\b", r"\banti-hero\b", r"\bcorrupt\w*\b",
        r"\bno good guys\b", r"\bdark choices\b", r"\bcompromise\b", r"\bvicious cycle\b"
    ],
    "Dystopian Bureaucracy": [
        r"\bdystop\w*\b", r"\btotalitarian\b", r"\bauthoritarian\b", r"\bsurveillance\b",
        r"\boppression\b", r"\bregime\b", r"\bsocietal collapse\b", r"\brebellion\b"
    ],
    "Coming of Age": [
        r"\bcoming of age\b", r"\badolescen\w*\b", r"\byouth\b", r"\bgrowing up\b",
        r"\bteenager\b", r"\bhigh school\b", r"\bloss of innocence\b", r"\bmaturation\b"
    ],
    "Revenge & Obsession": [
        r"\brevenge\b", r"\bvengeance\b", r"\bobsession\b", r"\bvow\b",
        r"\breckoning\b", r"\bhunt down\b", r"\bvendetta\b", r"\bpayback\b"
    ],
    "Cyberpunk & AI": [
        r"\bcyberpunk\b", r"\bcyborg\b", r"\bartificial intelligence\b", r"\bvirtual reality\b",
        r"\bhacker\b", r"\bneon\b", r"\bandroid\b", r"\bmegacorporation\b", r"\bneural\b"
    ],
    "Post-Apocalyptic": [
        r"\bpost-apocalyptic\b", r"\bapocalypse\b", r"\bwasteland\b", r"\bsurvival\b",
        r"\bruined world\b", r"\bextinction\b", r"\bfall of civilization\b"
    ]
}

def tag_text(text: str, genres: str = ""):
    text_lower = f"{genres} {text}".lower()
    matched = []
    for tag, patterns in VIBE_RULES.items():
        count = 0
        for pat in patterns:
            if re.search(pat, text_lower):
                count += 1
        if count > 0:
            matched.append((tag, count))
    
    # Sort by number of matched patterns descending
    matched.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in matched[:4]]
