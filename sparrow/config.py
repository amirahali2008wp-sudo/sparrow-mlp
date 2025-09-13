# sparrow/config.py

class SparrowConfig:
    """
    کلاس تنظیمات برای کنترل تمام هایپرپارامترهای معماری DynamicMLP.
    """
    def __init__(
        self,
        # پارامترهای اصلی معماری
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        num_experts: int = 8,
        expert_hidden_size: int = 128,
        
        # پارامترهای مسیریاب سراسری (کنترل لایه‌ها)
        layer_sparsity_lambda: float = 0.01,
        
        # پارامترهای مسیریاب محلی (کنترل متخصصان)
        expert_embedding_dim: int = 32,
        load_balancing_alpha: float = 0.01,
        entropy_lambda: float = 0.0  # به طور پیش‌فرض غیرفعال
    ):
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_experts = num_experts
        self.expert_hidden_size = expert_hidden_size
        self.layer_sparsity_lambda = layer_sparsity_lambda
        self.expert_embedding_dim = expert_embedding_dim
        self.load_balancing_alpha = load_balancing_alpha
        self.entropy_lambda = entropy_lambda