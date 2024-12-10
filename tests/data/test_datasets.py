from tatm.data.datasets import TatmCaptionedImageDataset

def test_captioned_image_dataset():
    ds = TatmCaptionedImageDataset("tests/data/multimodal", ["tests/data/multimodal/annotations.json"], 
                                   img_processor=lambda x: x, 
                                   text_processor=lambda x: x)
    
    assert len(ds) == 2

    first = ds[0]
    assert first is not None
    assert first["image"] is not None
    assert first["caption"] == "Jayson Tatum taking a shot."
    assert first["image_id"] == "test_image_1"

    second = ds[1]
    assert second is None
