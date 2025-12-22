\
import regex as re

DANMU_PATTERNS = [
    ("ritual_call", re.compile(r"(恭迎|恭贺|观礼|合影|结婴|天尊|韩老魔|韩天尊)")),
    ("viewing_status", re.compile(r"(来了|开始了|我准备好了|二刷|三刷|n刷|补番|回来了|前排|打卡)")),
    ("role_label", re.compile(r"(韩老魔|韩天尊|老魔|天尊|天使投资人)")),
    ("quote_ritual", re.compile(r"(道友|诸位|此番|在下|谨以|名台词|诗|词)")),
]

def tag_danmu_func(text: str, emo: str) -> str:
    t = text or ""
    for tag, pat in DANMU_PATTERNS:
        if pat.search(t):
            return tag
    if emo == "touching":
        return "emo_touching"
    if emo in ("praise", "laugh"):
        return "emo_like"
    return "other"

COMMENT_PATTERNS = [
    ("ritual_call", re.compile(r"(恭贺|观礼|散修观礼|结婴|天尊|韩老魔|韩天尊)")),
    ("info_comment", re.compile(r"(原著|小说|设定|其实|补充|解释|这里|剧情|伏笔|细节|对照)")),
    ("promotion_meta", re.compile(r"(三连|投币|点赞|收藏|关注|周边|活动|打call|支持|冲)")),
    ("qa", re.compile(r"(\?|？|求解|为啥|为什么|啥意思|怎么看)")),
]
TOUCHING_STORY = re.compile(r"(我|本人|当年|以前|那年|后来|陪伴|失恋|分手|工作|毕业|家里|爸妈|孩子|一路|经历)")

def tag_comment_func(text: str, emo: str) -> str:
    t = text or ""
    for tag, pat in COMMENT_PATTERNS:
        if pat.search(t):
            return tag
    if emo == "touching" and TOUCHING_STORY.search(t):
        return "touching_story"
    return "other"
