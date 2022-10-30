from microbundlecompute import image_analysis as ia


def test_hello_world():
    res = ia.hello_microbundle_compute()
    assert res == "Hello World!"
