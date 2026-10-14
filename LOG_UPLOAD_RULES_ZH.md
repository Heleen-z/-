# 公司软件运行日志 LOG 接入规则

适用范围：各软件向 AI Portal 的 LOG 列表上传运行记录。LOG 使用现有九个业务列。本文规定上传信息及映射，不代表已修改软件或已完成线上接口联调。

## 1. 列表地址与名称

- 站点地址：`https://faurecia.sharepoint.com/sites/AIPortal`
- 列表名称：`LOG`
- 列表浏览地址：`https://faurecia.sharepoint.com/sites/AIPortal/Lists/LOG/AllItems.aspx`
- 新增记录 REST 地址：`https://faurecia.sharepoint.com/sites/AIPortal/_api/web/lists/getbytitle('LOG')/items`
- 站点、列表目标及工具身份由各软件的受控配置维护。

## 2. 显示名称、内部名与上传键

显示名称是网页上看到的列名。内部名是列创建时确定的标识。更改显示名称后，程序仍使用原内部名对应的 REST 属性写入。字段元数据中的 `EntityPropertyName` 是 REST 列表项属性名；对于本文这些普通英文列名，它通常与 `InternalName` 相同，接入时以实际元数据为准。

用户将默认“标题”列改名为“工具名称”时，对应关系是：

```text
网页显示名称：工具名称
内部名：Title
上传 JSON 键：Title
上传的值：软件的完整显示名称
```

下面的映射依据当前列设置截图和建列过程整理。已确认的是截图中的显示名称与类型；其余英文列的内部名按直接使用这些英文名称创建的情况列出，首次接入需通过第 5 节核对。

| 网页列名 | 预期内部名 / 上传键 | 列类型 | 上传的信息 | 示例 |
| --- | --- | --- | --- | --- |
| 工具名称 | `Title` | 单行文本 | 软件显示名称，由当前软件配置提供 | `Testing Data Consolidation Tool` |
| EventId | `EventId` | 单行文本 | 本次执行的 UUID；一次执行生成一次，上传重试复用 | `d32cb988-a768-4cef-86d8-796373fe44ea` |
| ToolId | `ToolId` | 单行文本 | 全公司分配给软件的固定代码；升级和显示名称变化不改变代码 | `TEST_DATA_CONSOLIDATION` |
| FeatureName | `FeatureName` | 单行文本 | 本次实际执行的功能代码 | `MergeData` |
| UserId | `UserId` | 单行文本 | 本次实际操作人的公司账号标识，优先使用 UPN/登录账号 | `example.user@company.example` |
| Status | `Status` | 选择 | 本次执行的最终结果；值必须属于现有列允许的选项 | `SUCCESS`（需存在于当前选项中） |
| Duration | `Duration` | 数字 | 本次功能执行耗时，单位为秒，传 JSON 数字 | `12.45` |
| TimeStamp | `TimeStamp` | 日期和时间 | 本次操作结束或失败时的时间，采用 UTC ISO 8601 | `2026-10-14T07:30:00Z` |
| AppVersion | `AppVersion` | 单行文本 | 实际执行本次操作的软件版本，按字符串发送 | `1.1.1` |

上传程序为九个业务字段提供明确值。SharePoint 现有的必填、唯一值、选项与精度设置保持现状；接入程序读取元数据确认兼容性。截图中的 Status 没有必填标记，程序仍主动填写最终结果。

## 3. 每个字段的采集规则

### 工具身份

`Title` 是显示名称；`ToolId` 是固定的软件身份。不同软件使用不同 ToolId，同一软件升级后继续沿用原 ToolId。Title、ToolId 从软件配置取得，不能用本次输入文件名或项目名称替代。

### 功能与事件

一条记录代表一次功能执行的最终结果。`FeatureName` 使用稳定功能代码，例如 MergeData、ConvertBOM、GenerateReport。功能的开始和结束边界由各软件接入说明定义。

当前数据合并工具的功能代码为 MergeData，边界为实际开始合并至结果保存和校验完成，或执行失败。

EventId 在本次操作开始时生成，成功和失败均沿用。日志重试、断网补传和上传通道切换都保持原 EventId；用户重新执行功能时生成新的 EventId。一次批次处理多个文件仍对应一次事件。

不能仅因有 EventId 就宣称服务端已防重。接入程序需要读取实际唯一约束，并结合查询确认处理不确定的上传结果；若列未强制唯一，“先查询再新增”仍不能保证并发下不重复。程序不得把所有 HTTP 400 当作已上传。

### 操作人

UserId 记录真实操作人的公司账号标识。优先使用公司 UPN/登录账号，不要求该字符串一定是可收邮件的邮箱地址。Windows 桌面软件可优先读取 `whoami /upn`；无法获取时使用 `DOMAIN\username` 形式，并保留来源前缀以避免不同域重名。

Web 应用使用已登录用户的企业身份。通过服务账号或队列代理上传时，UserId 仍保留实际操作人；SharePoint 系统“创建者”则由实际写入账号决定。对于无人值守任务，使用实际任务身份，例如 `SERVICE:scheduled-report`，统计时与人员身份区分。

完全无法获取身份时，事件先保留待补充状态，不伪造员工账号，也不统一填入一个会被统计为真实员工的默认值。各软件应保持身份来源稳定；UPN 与 DOMAIN\username 需要统一映射后才能做可靠的跨软件员工去重。

### 执行结果

Status 必须匹配 LOG 现有 Choice 选项的准确文本，首次接入读取 Choices。截图仅能确认类型为“选项”，不能确认具体选项清单。

如果当前已配置 SUCCESS、PARTIAL、FAILED，则统一解释为：

- SUCCESS：本次功能按约定完成。
- PARTIAL：完成部分工作并产生可用结果，适用于支持部分完成的软件。
- FAILED：本次功能未达到约定完成条件。

如果现有选项只有其他值，接入适配器以实际选项建立明确映射，不自行增加列表选项或发送不存在的值。上传失败属于传输状态，不得把业务上已完成的 SUCCESS 改成 FAILED；后续补传保留原业务结果。

### 耗时、时间和版本

- Duration 统一以秒为单位，使用单调时钟测量，传非负有限数字，建议在上传前保留两位小数。实际列表显示精度以现有设置为准。排除日志上传、排队和重试耗时。
- TimeStamp 为操作结束或失败时的 UTC 时间。例：`2026-10-14T07:30:00Z` 对应北京时间 2026-10-14 15:30:00。网页显示由 SharePoint 区域设置决定。
- 补传保持原 TimeStamp，不能用补传时间覆盖。
- AppVersion 从实际运行程序读取，成功和失败都提供。VBA/脚本类软件可以在发布配置或版本常量中维护自身版本。
- 软件版本是字符串，例如 `1.10.0`，不能转换为浮点数字。补传时保留事件发生时版本，不能替换成补传程序当前版本。

## 4. 系统字段

以下列由 SharePoint 维护，不加入九字段业务上传正文：

| 网页名称 | 通常的内部名 | 含义 |
| --- | --- | --- |
| ID | `ID` | SharePoint 给列表项分配的编号，新增成功后读取 |
| 创建时间 | `Created` | 本条记录实际写入 SharePoint 的时间 |
| 修改时间 | `Modified` | 本条记录在 SharePoint 最近一次修改的时间 |
| 创建者 | `Author` | 实际写入该记录的账号 |
| 修改者 | `Editor` | 最近一次修改记录的账号 |

UserId 是事件里的操作人；Author 是 SharePoint 写入者。TimeStamp 是业务操作结束时间；Created 是服务器收到记录的时间。通过代理上传或离线补传时，这些值可能不同。

## 5. 如何核对内部名

### 从列设置地址查看

1. 打开 LOG 的列表设置。
2. 点击“工具名称”或其他列。
3. 查看浏览器地址里的 `Field=` 参数。
4. `Field=Title` 说明内部名为 Title；`Field=TimeStamp` 说明内部名为 TimeStamp。
5. 如参数经过 URL 编码，先进行 URL 解码；SharePoint 名称本身的 `_x0020_` 等编码保留，程序使用实际属性名。

此过程只查看，不需要修改列表。`TimeStamp`、`Timestamp` 的大小写以元数据返回值为准，上传键不凭显示名称重新猜测。

### 从 REST 元数据查看

用有权限的公司账号登录 SharePoint 后，读取：

```http
GET https://faurecia.sharepoint.com/sites/AIPortal/_api/web/lists/getbytitle('LOG')/fields?$select=Title,InternalName,EntityPropertyName,TypeAsString,Required,EnforceUniqueValues&$filter=Hidden%20eq%20false
Accept: application/json;odata=nometadata
```

对于默认标题改名后的字段，预期会得到类似结果：

```json
{
  "Title": "工具名称",
  "InternalName": "Title",
  "EntityPropertyName": "Title",
  "TypeAsString": "Text"
}
```

注意：这个字段元数据对象里的 Title 表示“列的显示名称”；新增列表项 JSON 里的 Title 则表示“给默认标题列写入的值”。两者所属对象不同。

核对 Status 的内部名后读取其选项，例如：

```http
GET https://faurecia.sharepoint.com/sites/AIPortal/_api/web/lists/getbytitle('LOG')/fields/getbyinternalnameortitle('Status')?$select=Title,InternalName,Choices,Required
Accept: application/json;odata=nometadata
```

普通浏览器直接打开 GET 地址时可能显示 XML；这是响应格式差异，不代表接口失败。程序按所需 Accept 头请求并解析。

## 6. 上传样例

以下为合成示例，假定实际元数据与上表一致，Status 已允许 SUCCESS。正文包含九个业务属性：

```json
{
  "Title": "Testing Data Consolidation Tool",
  "EventId": "d32cb988-a768-4cef-86d8-796373fe44ea",
  "ToolId": "TEST_DATA_CONSOLIDATION",
  "FeatureName": "MergeData",
  "UserId": "example.user@company.example",
  "Status": "SUCCESS",
  "Duration": 12.45,
  "TimeStamp": "2026-10-14T07:30:00Z",
  "AppVersion": "1.1.1"
}
```

该 JSON 是业务数据，不包含完整认证请求。上传客户端负责使用已批准的认证方式，获取所需的请求摘要，并向 LOG 的 `/items` 地址 POST。

如果沿用当前项目的 `application/json;odata=verbose` 客户端，应先读取 LOG 的 `ListItemEntityTypeFullName`，再添加协议需要的 `__metadata.type`。该元数据不是业务列，不计入九字段；不要从列表显示名猜测或硬编码实体类型。

## 7. 数据合并工具接入示例

| 当前代码数据 | 新上传数据 |
| --- | --- |
| 配置中的 tool_name / 事件中的 Toolname | `Title` |
| 原 Title 中的执行结果 | `Status`，按现有 Choice 映射 |
| 现有 EventId | `EventId` |
| 固定软件代码配置 | `ToolId` |
| 当前合并功能代码 MergeData | `FeatureName` |
| 当前获取的公司 UPN；必要时域账号 | `UserId` |
| 当前测量的执行秒数 | `Duration` |
| 原 Timestamp 中的业务时间 | `TimeStamp`，以实际内部名核对为准 |
| 当前程序 __version__ | `AppVersion`，在所有终态事件中采集 |

实际软件改造需同时调整事件生成和上传字段映射，使每条终态事件在本地保存后即可按本规范上传。上传成功以服务器返回有效列表项编号或查证同一事件已存在为依据。

## 8. 参考

- [Microsoft：Field element，内部名与显示名称](https://learn.microsoft.com/en-us/sharepoint/dev/schema/field-element-list)
- [Microsoft：字段 EntityPropertyName](https://learn.microsoft.com/en-us/previous-versions/office/sharepoint-visio/jj838485%28v%3Doffice.15%29)
- [Microsoft：使用 REST 操作列表和列表项](https://learn.microsoft.com/en-us/sharepoint/dev/sp-add-ins/working-with-lists-and-list-items-with-rest)
