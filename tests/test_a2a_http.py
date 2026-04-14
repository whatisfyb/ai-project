"""A2A HTTP Transport 测试脚本

在同一进程中启动一个 Server Agent 和一个 Client，
测试 HTTP 通信是否正常工作。
"""

import threading
import time
import sys

# 添加项目路径
sys.path.insert(0, ".")


def run_test():
    from agent.a2a import (
        A2AClient,
        A2AServer,
        InMemoryTransport,
        AgentCard,
        AgentCapabilities,
        Skill,
        Task,
        Message,
        TaskStatus,
    )

    print("=" * 50)
    print("A2A HTTP Transport 测试")
    print("=" * 50)

    # 1. 创建 Agent Card
    card = AgentCard(
        id="test-agent",
        name="测试 Agent",
        description="用于测试 HTTP Transport 的示例 Agent",
        capabilities=AgentCapabilities(
            text=True,
            files=False,
            streaming=False,
            push_notifications=False,
        ),
        skills=[
            Skill(name="echo", description="回显消息"),
        ],
    )

    # 2. 创建 Transport
    transport = InMemoryTransport()

    # 3. 定义消息处理器
    received_messages = []

    def handler(task: Task, message: Message):
        print(f"[Server] 收到消息: {message.get_text()}")
        received_messages.append(message.get_text())

        # 模拟处理：回显消息
        response = Message.agent_text(f"Echo: {message.get_text()}")
        transport.add_task_message(task.id, response)
        transport.update_task_status(task.id, TaskStatus.COMPLETED)

    # 4. 启动 Server
    server = A2AServer(
        agent_id="test-agent",
        card=card,
        transport=transport,
        handler=handler,
        port=8765,
    )

    print("\n[1] 启动 A2A Server...")
    server.start()
    time.sleep(1)  # 等待服务器启动

    print(f"[Server] 已启动，监听端口 8765")

    # 5. 创建 Client
    print("\n[2] 创建 A2A Client...")
    client = A2AClient("http://localhost:8765")

    # 6. 测试 get_card
    print("\n[3] 测试 agent/getCard...")
    try:
        remote_card = client.get_card()
        print(f"[Client] 获取到 Agent Card:")
        print(f"  - ID: {remote_card.id}")
        print(f"  - Name: {remote_card.name}")
        print(f"  - Skills: {[s.name for s in remote_card.skills]}")
    except Exception as e:
        print(f"[Client] 获取 Agent Card 失败: {e}")
        print("\n测试失败：无法连接到 Server")
        return False

    # 7. 测试 message_send
    print("\n[4] 测试 message/send...")
    try:
        # 创建消息
        test_message = Message.user_text("Hello, A2A!")

        # 发送消息
        task = client.message_send("test-task-001", test_message)
        print(f"[Client] 发送消息成功，Task ID: {task.id}")
        print(f"[Client] Task 状态: {task.status.value}")

        # 等待处理完成
        time.sleep(1)

        # 查询任务状态
        updated_task = client.tasks_get(task.id)
        if updated_task:
            print(f"[Client] 任务状态更新: {updated_task.status.value}")
            if updated_task.history:
                last_msg = updated_task.history[-1]
                print(f"[Client] 最后响应: {last_msg.get_text()}")

    except Exception as e:
        print(f"[Client] 消息发送失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 8. 验证结果
    print("\n[5] 验证结果...")
    if received_messages and "Hello, A2A!" in received_messages[0]:
        print("✓ 服务器收到了正确的消息")
        print("✓ 客户端成功获取了 Agent Card")
        print("✓ HTTP Transport 测试通过！")
        return True
    else:
        print("✗ 测试失败")
        return False


if __name__ == "__main__":
    try:
        success = run_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
