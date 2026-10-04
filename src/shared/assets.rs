use std::{collections, sync};

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use strum::{EnumString, VariantArray};

include!(concat!(env!("OUT_DIR"), "/assets.rs"));

pub(crate) static CHARACTERS_IDS: sync::LazyLock<Vec<&str>> = sync::LazyLock::new(|| {
    vec![
        "none",
        "endministratorm",
        "endministratorf",
        "perlica",
        "chenqy",
        "wulfgard",
        "ikut",
        "azrila",
        "seraph",
        "avywen",
        "aglina",
        "aurora",
        "lifeng",
        "laevatain",
        "yvonne",
        "dapan",
        "karin",
        "catcher",
        "estella",
        "bounda",
        "antal",
        "deepfin",
        "ardelia",
        "lastrite",
        "tangtang",
        "wulfa",
        "pograni",
        "zhuangfy",
        "mifu",
        "lzy",
        "camille",
        "typhoea",
        "liino",
        "purrche",
        "angelu",
        "liushuyun",
        "spl_fiona",
        "qinjc",
        "andrew",
        "aliya",
        "kelao",
        "muhui",
        "hanfang",
        "buyuan",
        "geyipu",
        "weixidong",
        "fabian",
        "ailaizha",
        "dannier",
        "hansi",
        "hateman",
        "helao",
        "hongbo",
        "lakuier",
        "lameng",
        "luoman",
        "madina",
        "nufuman",
        "shenjiaoe",
        "suosi",
        "swordmaster",
        "wuduanxue",
        "xiaona",
        "yalishanda",
    ]
});

pub(crate) static CHARACTERS_AVATARS: sync::LazyLock<collections::BTreeMap<&str, Asset>> = sync::LazyLock::new(|| {
    let mut map = collections::BTreeMap::default();

    map.insert("endministratorm", ICON_ROUND_CHR_0002_ENDMINM);
    map.insert("endministratorf", ICON_ROUND_CHR_0003_ENDMINF);
    map.insert("perlica", ICON_ROUND_CHR_0004_PELICA);
    map.insert("chenqy", ICON_ROUND_CHR_0005_CHEN);
    map.insert("wulfgard", ICON_ROUND_CHR_0006_WOLFGD);
    map.insert("ikut", ICON_ROUND_CHR_0007_IKUT);
    map.insert("azrila", ICON_ROUND_CHR_0009_AZRILA);
    map.insert("seraph", ICON_ROUND_CHR_0011_SERAPH);
    map.insert("avywen", ICON_ROUND_CHR_0012_AVYWEN);
    map.insert("aglina", ICON_ROUND_CHR_0013_AGLINA);
    map.insert("aurora", ICON_ROUND_CHR_0014_AURORA);
    map.insert("lifeng", ICON_ROUND_CHR_0015_LIFENG);
    map.insert("laevatain", ICON_ROUND_CHR_0016_LAEVAT);
    map.insert("yvonne", ICON_ROUND_CHR_0017_YVONNE);
    map.insert("dapan", ICON_ROUND_CHR_0018_DAPAN);
    map.insert("karin", ICON_ROUND_CHR_0019_KARIN);
    map.insert("catcher", ICON_ROUND_CHR_0020_MEURS);
    map.insert("estella", ICON_ROUND_CHR_0021_WHITEN);
    map.insert("bounda", ICON_ROUND_CHR_0022_BOUNDA);
    map.insert("antal", ICON_ROUND_CHR_0023_ANTAL);
    map.insert("deepfin", ICON_ROUND_CHR_0024_DEEPFIN);
    map.insert("ardelia", ICON_ROUND_CHR_0025_ARDELIA);
    map.insert("lastrite", ICON_ROUND_CHR_0026_LASTRITE);
    map.insert("tangtang", ICON_ROUND_CHR_0027_TANGTANG);
    map.insert("wulfa", ICON_ROUND_CHR_0028_WULFA);
    map.insert("pograni", ICON_ROUND_CHR_0029_POGRANI);
    map.insert("zhuangfy", ICON_ROUND_CHR_0030_ZHUANGFY);
    map.insert("mifu", ICON_ROUND_CHR_0031_MIFU);
    map.insert("lzy", ICON_ROUND_CHR_0032_LIZHIYAN);
    map.insert("camille", ICON_ROUND_CHR_0033_CAMILLE);
    map.insert("typhoea", ICON_ROUND_CHR_0034_TYPHOEA);
    map.insert("liino", ICON_ROUND_CHR_0035_LIINO);
    map.insert("purrche", ICON_ROUND_CHR_0038_PURRCHE);

    map.insert("angelu", ICON_SNS_NPC_ANGELU_01);
    map.insert("liushuyun", ICON_SNS_NPC_LIUSHUYUN_01);
    map.insert("spl_fiona", ICON_SNS_NPC_SPL_FIONA_01);
    map.insert("qinjc", ICON_SNS_NPC_QINJC_01);
    map.insert("andrew", ICON_SNS_NPC_ANDREW_01);
    map.insert("aliya", ICON_SNS_NPC_ALIYA_01);
    map.insert("kelao", ICON_SNS_NPC_KELAO_01);
    map.insert("muhui", ICON_SNS_NPC_MUHUI_01);
    map.insert("hanfang", ICON_SNS_NPC_HANFANG_01);
    map.insert("buyuan", ICON_SNS_NPC_BUYUAN_01);
    map.insert("geyipu", ICON_SNS_NPC_GEYIPU_01);
    map.insert("weixidong", ICON_SNS_NPC_WEIXIDONG_01);
    map.insert("fabian", ICON_SNS_NPC_FABIAN_01);
    map.insert("ailaizha", ICON_SNS_NPC_AILAIZHA_01);
    map.insert("dannier", ICON_SNS_NPC_DANNIER_01);
    map.insert("hansi", ICON_SNS_NPC_HANSI_01);
    map.insert("hateman", ICON_SNS_NPC_HATEMAN_01);
    map.insert("helao", ICON_SNS_NPC_HELAO_01);
    map.insert("hongbo", ICON_SNS_NPC_HONGBO_01);
    map.insert("lakuier", ICON_SNS_NPC_LAKUIER_01);
    map.insert("lameng", ICON_SNS_NPC_LAMENG_01);
    map.insert("luoman", ICON_SNS_NPC_LUOMAN_01);
    map.insert("madina", ICON_SNS_NPC_MADINA_01);
    map.insert("nufuman", ICON_SNS_NPC_NUFUMAN_01);
    map.insert("shenjiaoe", ICON_SNS_NPC_SHENJIAOE_01);
    map.insert("suosi", ICON_SNS_NPC_SUOSI_01);
    map.insert("swordmaster", ICON_SNS_NPC_SWORDMASTER_01);
    map.insert("wuduanxue", ICON_SNS_NPC_WUDUANXUE_01);
    map.insert("xiaona", ICON_SNS_NPC_XIAONA_01);
    map.insert("yalishanda", ICON_SNS_NPC_YALISHANDA_01);

    map.insert("none", ICON_SNS_NPC_SINGLE);

    map
});

pub(crate) static CHARACTERS_NAME: sync::LazyLock<collections::BTreeMap<&str, &str>> = sync::LazyLock::new(|| {
    let mut map = collections::BTreeMap::default();

    map.insert("endministratorm", "管理员 - M");
    map.insert("endministratorf", "管理员 - F");
    map.insert("perlica", "佩丽卡");
    map.insert("chenqy", "陈千语");
    map.insert("wulfgard", "狼卫");
    map.insert("ikut", "弧光");
    map.insert("azrila", "余烬");
    map.insert("seraph", "赛希");
    map.insert("avywen", "艾维文娜");
    map.insert("aglina", "洁尔佩塔");
    map.insert("aurora", "昼雪");
    map.insert("lifeng", "黎风");
    map.insert("laevatain", "莱万汀");
    map.insert("yvonne", "伊冯");
    map.insert("dapan", "大潘");
    map.insert("karin", "秋栗");
    map.insert("catcher", "卡契尔");
    map.insert("estella", "埃特拉");
    map.insert("bounda", "萤石");
    map.insert("antal", "安塔尔");
    map.insert("deepfin", "阿列什");
    map.insert("ardelia", "艾尔黛拉");
    map.insert("lastrite", "别礼");
    map.insert("tangtang", "汤汤");
    map.insert("wulfa", "洛茜");
    map.insert("pograni", "骏卫");
    map.insert("zhuangfy", "庄方宜");
    map.insert("mifu", "弭弗");
    map.insert("lzy", "李织烟");
    map.insert("camille", "卡缪");
    map.insert("typhoea", "提弗洛斯");
    map.insert("liino", "梨诺");
    map.insert("purrche", "噗切娜");

    map.insert("angelu", "[NPC] 安格鲁");
    map.insert("liushuyun", "[NPC] 盈");
    map.insert("spl_fiona", "[NPC] 联络员菲奥娜");
    map.insert("qinjc", "[NPC] 秦茳尺");
    map.insert("andrew", "[NPC] 安德烈");
    map.insert("aliya", "[NPC] 阿丽娅");
    map.insert("kelao", "[NPC] 克劳");
    map.insert("muhui", "[NPC] 青芜");
    map.insert("hanfang", "[NPC] 韩方");
    map.insert("buyuan", "[NPC] 卜圆");
    map.insert("geyipu", "[NPC] 葛一朴");
    map.insert("weixidong", "[NPC] 卫西东");
    map.insert("fabian", "[NPC] 法比安·柯林斯");
    map.insert("ailaizha", "[NPC] 艾莱扎·柯林斯");
    map.insert("dannier", "[NPC] 丹尼尔");
    map.insert("hansi", "[NPC] hansi");
    map.insert("hateman", "[NPC] hateman");
    map.insert("helao", "[NPC] 河佬");
    map.insert("hongbo", "[NPC] 洪波");
    map.insert("lakuier", "[NPC] lakuier");
    map.insert("lameng", "[NPC] lameng");
    map.insert("luoman", "[NPC] 罗曼");
    map.insert("madina", "[NPC] madina");
    map.insert("nufuman", "[NPC] 诺夫曼");
    map.insert("shenjiaoe", "[NPC] shenjiaoe");
    map.insert("suosi", "[NPC] 索斯");
    map.insert("swordmaster", "[NPC] swordmaster");
    map.insert("wuduanxue", "[NPC] wuduanxue");
    map.insert("xiaona", "[NPC] xiaona");
    map.insert("yalishanda", "[NPC] yalishanda");

    map.insert("none", "未知");

    map
});

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize, EnumString, VariantArray)]
#[strum(serialize_all = "lowercase")]
pub(crate) enum Emoji {
    Smile,
    StarEye,
    Surprise,
    Smug,
    Thumb,
    Please,
    Lol,
    Cry,
    Unwell,
    Sweat,
    Cool,
    Playful,
    Confused,
    Sorry,
    Pray,
    Ok,
    Tongue,
    Love,
    Blush,
    Joy,
    Heart,
    Sparkle,
    SweatSmile,
    Laugh,
    PlusOne,
    SideEye,
    Hundred,
    Dead,
    Anger,
    Angry,
    Dizzy,
    Scream,
    Sad,
    Sleep,
    Doubt,
    Speechless,
    FistBump,
    Think,
}

impl From<Emoji> for Asset {
    fn from(value: Emoji) -> Self {
        match value {
            Emoji::Smile => SNS_EMOJI_001,
            Emoji::StarEye => SNS_EMOJI_002,
            Emoji::Surprise => SNS_EMOJI_003,
            Emoji::Smug => SNS_EMOJI_004,
            Emoji::Thumb => SNS_EMOJI_005,
            Emoji::Please => SNS_EMOJI_006,
            Emoji::Lol => SNS_EMOJI_007,
            Emoji::Cry => SNS_EMOJI_008,
            Emoji::Unwell => SNS_EMOJI_009,
            Emoji::Sweat => SNS_EMOJI_010,
            Emoji::Cool => SNS_EMOJI_011,
            Emoji::Playful => SNS_EMOJI_012,
            Emoji::Confused => SNS_EMOJI_013,
            Emoji::Sorry => SNS_EMOJI_014,
            Emoji::Pray => SNS_EMOJI_015,
            Emoji::Ok => SNS_EMOJI_016,
            Emoji::Tongue => SNS_EMOJI_017,
            Emoji::Love => SNS_EMOJI_018,
            Emoji::Blush => SNS_EMOJI_019,
            Emoji::Joy => SNS_EMOJI_020,
            Emoji::Heart => SNS_EMOJI_021,
            Emoji::Sparkle => SNS_EMOJI_022,
            Emoji::SweatSmile => SNS_EMOJI_023,
            Emoji::Laugh => SNS_EMOJI_024,
            Emoji::PlusOne => SNS_EMOJI_025,
            Emoji::SideEye => SNS_EMOJI_026,
            Emoji::Hundred => SNS_EMOJI_027,
            Emoji::Dead => SNS_EMOJI_028,
            Emoji::Anger => SNS_EMOJI_029,
            Emoji::Angry => SNS_EMOJI_030,
            Emoji::Dizzy => SNS_EMOJI_031,
            Emoji::Scream => SNS_EMOJI_032,
            Emoji::Sad => SNS_EMOJI_033,
            Emoji::Sleep => SNS_EMOJI_034,
            Emoji::Doubt => SNS_EMOJI_035,
            Emoji::Speechless => SNS_EMOJI_036,
            Emoji::FistBump => SNS_EMOJI_037,
            Emoji::Think => SNS_EMOJI_038,
        }
    }
}
