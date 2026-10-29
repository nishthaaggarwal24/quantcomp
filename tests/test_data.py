from src.data import LABELS, load_frame, make_split

def test_dataset_deduplicated_and_expected_size():
    df=load_frame()
    assert len(df)==388
    assert set(df.label)==set(LABELS)
    assert not df.duplicated(["label","text"]).any()

def test_split_is_stratified_seeded_and_disjoint():
    a=make_split(2); b=make_split(2)
    assert (a.y_train==b.y_train).all() and (a.y_test==b.y_test).all()
    assert len(a.y_train)==291 and len(a.y_test)==97
    assert set(a.y_train)==set(range(8)) and set(a.y_test)==set(range(8))
    assert a.X_train.shape==(291,8) and a.X_test.shape==(97,8)
    assert (a.X_train.min()>=0) and (a.X_train.max()<=3.141593)
