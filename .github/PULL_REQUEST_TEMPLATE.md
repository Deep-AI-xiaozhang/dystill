# Pull Request

## 这个 PR 做了什么

<!-- 一两句话说清动机和结果。关联 issue 请写 Closes #123。 -->

## 类型

- [ ] Bug 修复
- [ ] 新功能
- [ ] 文档 / 注释
- [ ] 重构（不改变行为）
- [ ] 打包 / CI

## 自测清单

- [ ] `pytest -q` 全绿
- [ ] `ruff check .` 全绿
- [ ] 新增/修改的行为有对应测试（或说明为什么不需要）
- [ ] 改了 `config.example.yaml` 时，已同步 `src/distill_workshop/config.example.yaml`（两份必须逐字节一致）
- [ ] 涉及打包/依赖的改动，已本地 `python -m build` 并验证 wheel 可安装、`dystill --version` 可用

## 隐私与仓库卫生

- [ ] 没有提交 `config.yaml`、`.env`、任何 API key / cookie / token
- [ ] 没有提交 `data/`、`cache/`、`bench/`、`outputs/` 或用户视频/笔记原文
- [ ] 日志、截图、测试夹具里没有真实凭据和私人链接

## 影响面

<!-- 是否改动默认行为、配置格式、缓存格式或 CLI 参数？需要迁移吗？ -->
