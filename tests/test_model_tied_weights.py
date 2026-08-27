from model.model import MokioMindConfig, MokioMindForCausalLM


def test_lm_head_and_input_embeddings_are_tied():
    config = MokioMindConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=1)
    model = MokioMindForCausalLM(config)

    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
