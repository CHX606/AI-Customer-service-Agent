RETRIEVAL_CASES = [
    {
        "id": "q001",
        "type": "clear",
        "question": (
            "套餐时间已经延长了，"
            "但是剩余流量还是没有恢复。"
        ),
        "expected_sections": ["1.2"],
        "expected_action": "answer",
    },
    {
        "id": "q002",
        "type": "clear",
        "question": (
            "钱已经支付成功了，但是账号里"
            "没有显示套餐，时间和流量也没增加。"
        ),
        "expected_sections": ["1.1"],
        "expected_action": "request_image",
    },
    {
        "id": "q003",
        "type": "clear",
        "question": (
            "我买错套餐了，能不能换成"
            "另外一个套餐？"
        ),
        "expected_sections": ["1.3"],
        "expected_action": "answer",
    },
    {
        "id": "q004",
        "type": "clear",
        "question": "购买的套餐可以退款吗？",
        "expected_sections": ["1.6"],
        "expected_action": "answer",
    },
    {
        "id": "q005",
        "type": "clear",
        "question": (
            "我知道注册邮箱，但是忘记了"
            "账号密码，应该怎么办？"
        ),
        "expected_sections": ["2"],
        "expected_action": "answer",
    },
    {
        "id": "q006",
        "type": "clear",
        "question": (
            "点击一键导入没有反应，"
            "订阅没有进入软件。"
        ),
        "expected_sections": ["4.3"],
        "expected_action": "answer",
    },
    {
        "id": "q007",
        "type": "clear",
        "question": (
            "订阅已经导入了，但是所有节点"
            "都显示超时，没有延迟数字。"
        ),
        "expected_sections": ["4.2"],
        "expected_action": "request_image",
    },
    {
        "id": "q008",
        "type": "clear",
        "question": (
            "iPhone上的Shadowrocket已经导入，"
            "但是节点连通性测试全部超时。"
        ),
        "expected_sections": ["4.5.1"],
        "expected_action": "request_image",
    },
    {
        "id": "q009",
        "type": "clear",
        "question": (
            "Windows上的Clash Verge看起来"
            "已经连接，但还是打不开网页。"
        ),
        "expected_sections": ["4.6"],
        "expected_action": "request_image",
    },
    {
        "id": "q010",
        "type": "clear",
        "question": (
            "这个月没有用完的套餐流量，"
            "可以留到下个月继续用吗？"
        ),
        "expected_sections": ["7"],
        "expected_action": "answer",
    },
    {
        "id": "q011",
        "type": "clear",
        "question": (
            "同一个套餐能不能同时在手机"
            "和Windows电脑上使用？"
        ),
        "expected_sections": ["7"],
        "expected_action": "answer",
    },
    {
        "id": "q012",
        "type": "clear",
        "question": (
            "使用香港节点访问AI服务时，"
            "为什么还是会触发平台风控？"
        ),
        "expected_sections": ["6"],
        "expected_action": "answer",
    },
    {
        "id": "q013",
        "type": "ambiguous",
        "question": "我已经买了，怎么还是用不了？",
        "expected_sections": [
            "4",
            "1.1",
            "4.3",
        ],
        "expected_action": "request_image",
    },
    {
        "id": "q014",
        "type": "ambiguous",
        "question": (
            "我刚付完钱，怎么还是"
            "什么变化都没有？"
        ),
        "expected_sections": [
            "1.1",
            "1.2",
        ],
        "expected_action": "request_image",
    },
    {
        "id": "q015",
        "type": "out_of_scope",
        "question": "北京明天会下雨吗？",
        "expected_sections": [],
        "expected_action": "out_of_scope",
    },
]