"""MATH 训练数据的答案修正。修正依据原题，使用前核对题目和解答的校验值。"""

from typing import NamedTuple


class GoldCorrection(NamedTuple):
    question_sha256: str
    solution_sha256: str
    answers: tuple[str, ...]
    reason: str


# 空answers是有记录的评分能力排除；多个answers表示任选其一，单个集合表达式要求全部成员。
MATH_GOLD_CORRECTIONS = {
    "train/algebra/row_334": GoldCorrection(
        "f03b90f3870b9bb19b331275c07cfa564903883542a6b5a528db711e0b402974",
        "f7c2dbcf5dd00afeac0131a4bccbfe0c4e4ff34699f0a852b6fe268eaac82e61",
        ("\\boxed{\\{(-2,18),(8,38)\\}}",),
        "全部交点；联立方程得到x=-2,8。",
    ),
    "train/algebra/row_504": GoldCorrection(
        "01b9fc53f80e069147f19eb75b78ea35d36ce151a1f9a2a2e691176f6264e729",
        "2d720bddff344d68d89af18141b2eb996728b3b2e1ff30bd0c9685841c1a3d31",
        ("\\boxed{\\{(-2,23),(0,1)\\}}",),
        "全部交点；联立方程得到x=-2,0。",
    ),
    "train/intermediate_algebra/row_75": GoldCorrection(
        "855b822dd34111db3c39215e9143866b9bb521ec2622ad78a0694c44b1b7d9c9",
        "56150cc25dd6508ce82040e3f6f865f1f8d79e3b483436b0dc4decc6ec45cf4e",
        ("\\boxed{\\{2x+1,-2x-1\\}}",),
        "多项式平方等于(2x+1)^2，正负两个均需要。",
    ),
    "train/intermediate_algebra/row_309": GoldCorrection(
        "7b3f2dd9edc7728a03734acd63d6b2b1d3bbd4df7ca9038c53b7861b40179f61",
        "4e2cf1c79be2dadce911855b4a8905a5dc6496ef40db037fec50e26925b49fa5",
        ("\\boxed{\\{\\frac{1}{3},-\\frac{5}{2}\\}}",),
        "全部根由(2t+5)(3t-1)=0得到。",
    ),
    "train/intermediate_algebra/row_612": GoldCorrection(
        "2e44c70d1b2c2aa66df3011fab326aa726577f63366a731ba6682b53092cf860",
        "db843ab7ce0876ab61b52852db191bcd20bea23f4670b23d1a0fd595628ea2b4",
        ("\\boxed{\\{1,9\\}}",),
        "原点到圆心距离5、半径4，内外切半径为1和9。",
    ),
    "train/intermediate_algebra/row_868": GoldCorrection(
        "ae287fa11863c08b79e6f7b936354fcb1847559e822a6a7fef861612782d558e",
        "6f439e03285e0630af9dab01341bdfa461039dc319450f1dcd471d8e108a6fe1",
        ("\\boxed{(-5,7)}", "\\boxed{(-5,1)}"),
        "原题明确只输入任意一个焦点；中心(-5,4)，焦距3。",
    ),
    "train/number_theory/row_766": GoldCorrection(
        "dc078797f4f3b5a6690bdb57c23585b82bc97487d722ba76465d8b2f9866124a",
        "f4cac71a576073b8945f2833966953a1e5fb4cc241340375acf751264ac3f3e7",
        (),
        "unsupported_clock_gold: 数学解析器将20:34和8:34 p.m.分别误读成10/17和4/17；暂不纳入训练或dev。",
    ),
    "train/precalculus/row_26": GoldCorrection(
        "1219800faf3e940eb77a849381967024967672eaf7d4feada21fa6fd41f05639",
        "d9f6c7e6d905e3f4cb016002142d5edf6bd70c2867692000c29b3a29b85eff3a",
        ("\\boxed{\\{12,18\\}}",),
        "k=6不满足，最小两个解为12和18。",
    ),
    "train/precalculus/row_62": GoldCorrection(
        "4974e964343df2a76e82fc7265d3b7b6f3884409dc1e33c46b548974f2d6a89e",
        "a3d5bfa69d8574be53fd4ced5ac707abd9ea532f9cc1fe7319eda9f5496c3260",
        ("\\boxed{\\{5,-3\\}}",),
        "特征方程(k-1)^2=16，两个解为5和-3。",
    ),
    "train/precalculus/row_89": GoldCorrection(
        "3d577c37405144270b0e79bb6a908fbd1cd4338c472ed921312d3c60051f92dc",
        "ad20ae36feb1ca4aa22f0477f99c4484e5c9f9c6c6228c4fca4f5ec24cd2fa7e",
        ("\\boxed{\\{30^\\circ,150^\\circ\\}}",),
        "cos(theta)=正负sqrt(3)/2，两个角为30和150度。",
    ),
    "train/precalculus/row_660": GoldCorrection(
        "77165c22df15d7a5f6f5f40da4dafaebdca5daf42504e4e021d44c5c0e70d0f8",
        "21ad1bb7c9f862d7c4fb8679f3741dec48fce461af291f060be898f6cb7859c2",
        ("\\boxed{\\{\\frac{3}{2},-3\\}}",),
        "行列式为0分成a+b+c=0或a=b=c；定义域内分别得到-3和3/2。",
    ),
    "train/precalculus/row_168": GoldCorrection(
        "1a2895aa61946e562550937ea71562995f2c7d41906e0453ece90d19de04d9b2",
        "0d7fa0e4cf72164ee7991b5453333ea647b4cd6acd8ff4eaded3814b049ab2d5",
        ("\\boxed{\\{45^\\circ,60^\\circ,75^\\circ\\}}",),
        "余弦定理得到45度和60度，内角和确定第三角75度，三个均需提交。",
    ),
    "train/precalculus/row_471": GoldCorrection(
        "9c4458f7810f197206c3f5644427a5d81c85f832104f302ad15ca7cab7a292ac",
        "b56d1a7541127855492b5c197bde9ee5e68f42110f7b8d1dc958b0537825f619",
        ("\\boxed{\\{-4,4\\}}",),
        "两个实解分别给出-4和4，原题要求全部可能值。",
    ),
    "train/precalculus/row_567": GoldCorrection(
        "182865d49dfe906b671b052b6f0f3eb66a7ea6e70c5c4821be35ff7df6ffb9f5",
        "53eabe7d03799fad4040a1e9212d3195744ec94591d3fdf26b55555a3609b5c0",
        (),
        "unsupported_answer_multiplicity: 三个内角为30,75,75；当前解析器将列表变为集合，丢失重数。",
    ),
    "train/intermediate_algebra/row_336": GoldCorrection(
        "fab859000e9e97afcbf007a100552a84b564eb89871f493cf18a69c82e6cadc3",
        "1f1c23ea062556a347370b67447adff2a3ffd82e24b40356f32fec619c070117",
        (),
        "unsupported_answer_multiplicity: 多项式为(x+1)(x-3)^2，原题明确要求重根重复输入；当前解析器丢失重数。",
    ),
    "train/precalculus/row_663": GoldCorrection(
        "b566fe7d1719df7ad87d51ce30bafd8744d2c6adb3119ea3990d05e734d1e082",
        "5b9691067e06470daa5b261bce07af1c89f0ac295809f46d5507437938f7cc5f",
        (
            "\\boxed{\\begin{pmatrix}2/3\\\\-2/3\\\\-1/3\\end{pmatrix}}",
            "\\boxed{\\begin{pmatrix}-2/3\\\\2/3\\\\1/3\\end{pmatrix}}",
        ),
        "叉积为(2,-2,-1)，范数3；任选一个正负单位法向量。",
    ),
    "train/precalculus/row_739": GoldCorrection(
        "c93ef863dfeed1931c1a461e9e69fd812f29ad08fca932b8b1f363b86e0b53d2",
        "c472554281034539d89caff611b3b433a549cbef91a33eb1343714401e945a6c",
        ("\\boxed{\\frac{\\pi}{3}}", "\\boxed{-\\frac{\\pi}{3}}"),
        "sin(3x)的周期为2pi/3；原解答明确接受正负pi/3两种方向的相移。",
    ),
    "train/number_theory/row_35": GoldCorrection(
        "38fe2db28af088f4428d0d9c09e50cfd1afd30fc7c361ab1304bbb819a332e74",
        "823bbd4c04efcb98a683582429e50912f7715b4da3e5d6ccfd92c58eb550f2a6",
        (),
        "unsupported_clock_gold: 4:51:06被误解析成(4/51)/6，无法可靠评分时分秒。",
    ),
    "train/prealgebra/row_599": GoldCorrection(
        "7443dcbc81608b149ddc9a6e94251bfbdafb942f22d969141bc4170815152588",
        "ede1c28eaf6a41b7362159b701eca22f569042019ff15d1fcdbfa5c6e653170e",
        (),
        "unsupported_clock_gold: 05:00被误解析成复无穷大，不能作数学表达式评分。",
    ),
    "train/prealgebra/row_800": GoldCorrection(
        "c93195e239617b199855e5450af0ae8885f2d52408c8b48621b5195ac013412e",
        "f0052f521f9158ae411ebd7d947682cf72adafb7fc29decd1021cd4e5fc1d760",
        (),
        "unsupported_clock_gold: 9:24被误解析成3/8，不能把时刻当比例。",
    ),
}
