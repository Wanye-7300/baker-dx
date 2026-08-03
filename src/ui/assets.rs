use std::sync;

use dioxus::prelude::*;

include!(concat!(env!("OUT_DIR"), "/assets.rs"));

pub(crate) static CHARACTERS_AVATARS: sync::LazyLock<fnv::FnvHashMap<&str, Asset>> = sync::LazyLock::new(|| {
    let mut hashmap = fnv::FnvHashMap::default();

    hashmap.insert("endministratorm", ICON_ROUND_CHR_0002_ENDMINM);
    hashmap.insert("endministratorf", ICON_ROUND_CHR_0003_ENDMINF);
    hashmap.insert("perlica", ICON_ROUND_CHR_0004_PELICA);
    hashmap.insert("chenqy", ICON_ROUND_CHR_0005_CHEN);
    hashmap.insert("wulfgard", ICON_ROUND_CHR_0006_WOLFGD);
    hashmap.insert("ikut", ICON_ROUND_CHR_0007_IKUT);
    hashmap.insert("azrila", ICON_ROUND_CHR_0009_AZRILA);
    hashmap.insert("seraph", ICON_ROUND_CHR_0011_SERAPH);
    hashmap.insert("avywen", ICON_ROUND_CHR_0012_AVYWEN);
    hashmap.insert("aglina", ICON_ROUND_CHR_0013_AGLINA);
    hashmap.insert("aurora", ICON_ROUND_CHR_0014_AURORA);
    hashmap.insert("lifeng", ICON_ROUND_CHR_0015_LIFENG);
    hashmap.insert("laevatain", ICON_ROUND_CHR_0016_LAEVAT);
    hashmap.insert("yvonne", ICON_ROUND_CHR_0017_YVONNE);
    hashmap.insert("dapan", ICON_ROUND_CHR_0018_DAPAN);
    hashmap.insert("karin", ICON_ROUND_CHR_0019_KARIN);
    hashmap.insert("catcher", ICON_ROUND_CHR_0020_MEURS);
    hashmap.insert("estella", ICON_ROUND_CHR_0021_WHITEN);
    hashmap.insert("bounda", ICON_ROUND_CHR_0022_BOUNDA);
    hashmap.insert("antal", ICON_ROUND_CHR_0023_ANTAL);
    hashmap.insert("deepfin", ICON_ROUND_CHR_0024_DEEPFIN);
    hashmap.insert("ardelia", ICON_ROUND_CHR_0025_ARDELIA);
    hashmap.insert("lastrite", ICON_ROUND_CHR_0026_LASTRITE);
    hashmap.insert("tangtang", ICON_ROUND_CHR_0027_TANGTANG);
    hashmap.insert("wulfa", ICON_ROUND_CHR_0028_WULFA);
    hashmap.insert("pograni", ICON_ROUND_CHR_0029_POGRANI);
    hashmap.insert("zhuangfy", ICON_ROUND_CHR_0030_ZHUANGFY);
    hashmap.insert("mifu", ICON_ROUND_CHR_0031_MIFU);
    hashmap.insert("lzy", ICON_ROUND_CHR_0032_LIZHIYAN);
    hashmap.insert("camille", ICON_ROUND_CHR_0033_CAMILLE);

    hashmap.insert("none", ICON_SNS_NPC_SINGLE);

    hashmap
});

pub(crate) static CHARACTERS_NAME: sync::LazyLock<fnv::FnvHashMap<&str, &str>> = sync::LazyLock::new(|| {
    let mut hashmap = fnv::FnvHashMap::default();

    hashmap.insert("endministratorm", "管理员 - M");
    hashmap.insert("endministratorf", "管理员 - F");
    hashmap.insert("perlica", "佩丽卡");
    hashmap.insert("chenqy", "陈千语");
    hashmap.insert("wulfgard", "狼卫");
    hashmap.insert("ikut", "弧光");
    hashmap.insert("azrila", "余烬");
    hashmap.insert("seraph", "赛希");
    hashmap.insert("avywen", "艾维文娜");
    hashmap.insert("aglina", "洁尔佩塔");
    hashmap.insert("aurora", "昼雪");
    hashmap.insert("lifeng", "黎风");
    hashmap.insert("laevatain", "莱万汀");
    hashmap.insert("yvonne", "伊冯");
    hashmap.insert("dapan", "大潘");
    hashmap.insert("karin", "秋栗");
    hashmap.insert("catcher", "卡契尔");
    hashmap.insert("estella", "埃特拉");
    hashmap.insert("bounda", "萤石");
    hashmap.insert("antal", "安塔尔");
    hashmap.insert("deepfin", "阿列什");
    hashmap.insert("ardelia", "艾尔黛拉");
    hashmap.insert("lastrite", "别礼");
    hashmap.insert("tangtang", "汤汤");
    hashmap.insert("wulfa", "洛茜");
    hashmap.insert("pograni", "骏卫");
    hashmap.insert("zhuangfy", "庄方宜");
    hashmap.insert("mifu", "弭弗");
    hashmap.insert("lzy", "李织烟");
    hashmap.insert("camille", "卡缪");

    hashmap.insert("none", "未知");

    hashmap
});

pub(crate) fn get_avatar(id: &str) -> Asset {
    CHARACTERS_AVATARS
        .get(id)
        .copied()
        .unwrap_or(CHARACTERS_AVATARS["none"])
}

pub(crate) fn get_group_avatar(id: &str) -> Asset {
    if id.is_empty() {
        ICON_ROUND_SNS_ENDFIELD_GROUP_A
    } else {
        get_avatar(id)
    }
}
