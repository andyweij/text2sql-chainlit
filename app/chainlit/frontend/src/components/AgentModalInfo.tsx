import * as React from "react";
import { useState } from "react"; // 修正 1: 引入 useState
import { useFormik } from "formik"; // 修正 1: 引入 useFormik
import * as yup from 'yup';
import { toast } from "sonner"; // 修正 1: 引入 toast

// Material-UI Components
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import ButtonGroup from "@mui/material/ButtonGroup";
import Typography from "@mui/material/Typography";
import Modal from "@mui/material/Modal";
import TextField from "@mui/material/TextField";

const style = {
  position: "absolute",
  top: "50%",
  left: "50%",
  transform: "translate(-50%, -50%)",
  width: 400,
  bgcolor: "background.paper",
  border: "2px solid #000",
  boxShadow: 24,
  p: 4,
  // Box 內部使用 flex 布局
  display: "flex",
  flexDirection: "column",
  gap: "12px", // 控制元件之間的垂直間距
};

// 表單驗證規則
const validationSchema = yup.object({
  agentName: yup.string().required('Agent Name 為必填項目'),
  llmModelName: yup.string().required('LLM Model Name 為必填項目'),
  endpoint: yup.string().url('請輸入有效的 URL').required('API Endpoint 為必填項目'),
  llmApiKey: yup.string().required('LLM API Key 為必填項目'),
  embeddingModalName: yup.string().required('Embedding Model Name 為必填項目'),
  embeddingEndpoint: yup.string().url('請輸入有效的 URL').required('Embedding Endpoint 為必填項目'),
  embeddingApiKey: yup.string().required('Embedding API Key 為必填項目'),
  dbConnectionString: yup.string().required('DB Connection String 為必填項目'),
});

interface AgentModalInfoProps {
  isOpen: boolean;
  onClose: () => void;
}


export const AgentModalInfo: React.FC<AgentModalInfoProps> = ({ isOpen, onClose }) => {
  const [isVerifying, setIsVerifying] = useState(false);


  const formik = useFormik({
    // 更新後的初始值，包含所有需要的欄位
    initialValues: {
      agentName: '',
      llmModelName: 'llama3.1-ffm-8b-32k-chat',
      endpoint: 'https://api-ams.twcc.ai/api/models',
      llmApiKey: '',
      embeddingModalName: 'ffm-embedding-v2.1',
      embeddingEndpoint: 'https://api-ams.twcc.ai/api/models',
      embeddingApiKey: '',
      dbConnectionString: '',
    },
  validationSchema: validationSchema,
  onSubmit: async (values, { setSubmitting }) => {
      setSubmitting(true);
      try {
        const response = await fetch('/api/create-agent', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(values),
        });

        if (response.ok) {
          toast.success(`Agent '${values.agentName}' 建立成功！`);
          onClose();
          // 重新載入頁面以刷新 Agent 列表
          window.location.reload();
        } else {
          const errorData = await response.json();
          toast.error(`建立失敗: ${errorData.detail || response.statusText}`);
        }
      } catch (error) {
        toast.error('發生網路錯誤，請稍後再試。');
        console.error('Failed to submit form:', error);
      } finally {
        setSubmitting(false);
      }
    },
  });

const handleVerifyConnection = async () => {
    setIsVerifying(true);

      // 從 formik 中提取需要驗證的資料
      const {
      endpoint,
      llmApiKey,
      llmModelName,
      embeddingEndpoint,
      embeddingApiKey,
      embeddingModalName,
      } = formik.values;
    try {
      // 假設為 OpenAI 相容的 API 端點，路徑通常是 /v1/chat/completions
      const llmResponse = await fetch(`${endpoint}/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${llmApiKey}`, // 使用 Bearer Token 進行授權
        },
        body: JSON.stringify({
          model: llmModelName, // 使用表單中指定的模型名稱
          messages: [{ role: 'user', content: 'Say hello' }], // 建立一個最小化的請求 Body
          max_tokens: 5,
        }),
      });

      if (llmResponse.ok) {
        toast.success('LLM Model 連線成功！');
      } else {
        const errorData = await llmResponse.json();
        // 提取 API 返回的具體錯誤訊息
        const errorMessage = errorData.error?.message || JSON.stringify(errorData);
        toast.error(`LLM Model 驗證失敗: ${errorMessage}`);
        setIsVerifying(false); // 驗證失敗，直接結束
        return;
      }
    } catch (error) {
      toast.error('LLM Model 網路請求失敗，請檢查 Endpoint URL 是否正確。');
      console.error('LLM connection test failed:', error);
      setIsVerifying(false); // 驗證失敗，直接結束
      return;
    }

    // --- 測試 2: 驗證 Embedding Model ---
    try {
      // 假設為 OpenAI 相容的 API 端點，路徑通常是 /v1/embeddings
      const embeddingResponse = await fetch(`${embeddingEndpoint}/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${embeddingApiKey}`,
        },
        body: JSON.stringify({
          model: embeddingModalName,
          input: 'This is a test.', // Embedding API 需要一個 input 字串
        }),
      });

      if (embeddingResponse.ok) {
        toast.success('Embedding Model 連線成功！');
      } else {
        const errorData = await embeddingResponse.json();
        const errorMessage = errorData.error?.message || JSON.stringify(errorData);
        toast.error(`Embedding Model 驗證失敗: ${errorMessage}`);
      }
    } catch (error) {
      toast.error('Embedding Model 網路請求失敗，請檢查 Endpoint URL 是否正確。');
      console.error('Embedding connection test failed:', error);
    } finally {
      setIsVerifying(false); // 所有驗證流程結束
    }
  };

  return (
    <div>
      <Modal
        open={isOpen}
        onClose={onClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <Box component="form" onSubmit={formik.handleSubmit} noValidate sx={style}>
          <Typography id="modal-modal-title" variant="h6" component="h2">
            新增Agent
          </Typography>

          <TextField
          id="agentName"
          name="agentName" // name 必須對應 initialValues
          label="Agent名稱"
          variant="outlined"
          fullWidth
          sx={{ mt: 1, mb: 2, display: "block" }}
          onChange={formik.handleChange}
          error={formik.touched.agentName && Boolean(formik.errors.agentName)}
          helperText={formik.touched.agentName && formik.errors.agentName}
          />

          <TextField
            margin="dense"
            id="llmModelName"
            name="llmModelName"
            label="語言模型"
            type="text"
            fullWidth
            variant="outlined"
            required
            value={formik.values.llmModelName}
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
            error={formik.touched.llmModelName && Boolean(formik.errors.llmModelName)}
            helperText={formik.touched.llmModelName && formik.errors.llmModelName}
            sx={{ mt: 0, mb: 0, display: "block" }}
          />

          {/* API Endpoint */}
          <TextField
            margin="dense"
            id="endpoint"
            name="endpoint"
            label="API端點"
            type="text"
            fullWidth
            variant="outlined"
            required
            value={formik.values.endpoint}
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
            error={formik.touched.endpoint && Boolean(formik.errors.endpoint)}
            helperText={formik.touched.endpoint && formik.errors.endpoint}
            sx={{ mt: 0, mb: 0, display: "block" }}
          />

          {/* LLM API Key */}
          <TextField
            margin="dense"
            id="llmApiKey"
            name="llmApiKey"
            label="API金鑰"
            type="password"
            fullWidth
            variant="outlined"
            required
            value={formik.values.llmApiKey}
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
            error={formik.touched.llmApiKey && Boolean(formik.errors.llmApiKey)}
            helperText={formik.touched.llmApiKey && formik.errors.llmApiKey}
            sx={{ mt: 0, mb: 2, display: "block" }}
          />

          {/* Embedding Model Name */}
          <TextField
            margin="dense"
            id="embeddingModalName"
            name="embeddingModalName"
            label="向量模型"
            type="text"
            fullWidth
            variant="outlined"
            required
            value={formik.values.embeddingModalName}
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
            error={formik.touched.embeddingModalName && Boolean(formik.errors.embeddingModalName)}
            sx={{ mt: 0, mb: 0, display: "block" }}
            helperText={formik.touched.embeddingModalName && formik.errors.embeddingModalName}
          />

          {/* Embedding Endpoint */}
          <TextField
            margin="dense"
            id="embeddingEndpoint"
            name="embeddingEndpoint"
            label="API端點"
            type="text"
            fullWidth
            variant="outlined"
            required
            value={formik.values.embeddingEndpoint}
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
            error={formik.touched.embeddingEndpoint && Boolean(formik.errors.embeddingEndpoint)}
            helperText={formik.touched.embeddingEndpoint && formik.errors.embeddingEndpoint}
            sx={{ mt: 0, mb: 0, display: "block" }}
          />

          {/* Embedding Api Key */}
          <TextField
            margin="dense"
            id="embeddingApiKey"
            name="embeddingApiKey"
            label="API金鑰"
            type="password"
            fullWidth
            variant="outlined"
            required
            value={formik.values.embeddingApiKey}
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
            error={formik.touched.embeddingApiKey && Boolean(formik.errors.embeddingApiKey)}
            helperText={formik.touched.embeddingApiKey && formik.errors.embeddingApiKey}
            sx={{ mt: 0, mb: 1, display: "block" }}
          />

          {/* DB Connection String */}
          <TextField
            margin="dense"
            id="dbConnectionString"
            name="dbConnectionString"
            label="資料庫URI"
            type="text"
            fullWidth
            variant="outlined"
            required
            value={formik.values.dbConnectionString}
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
            error={formik.touched.dbConnectionString && Boolean(formik.errors.dbConnectionString)}
            helperText={formik.touched.dbConnectionString && formik.errors.dbConnectionString}
            sx={{ mt: 0, mb: 1, display: "block" }}
          />

          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              "& > *": {
                m: 1,
              },
            }}
          >
            <ButtonGroup size="large" aria-label="Large button group">
               <Button onClick={onClose}>
                取消
              </Button>
              <Button
                onClick={handleVerifyConnection}
                disabled={isVerifying || formik.isSubmitting}
              >
                {isVerifying ? '驗證中...' : '驗證'}
              </Button>
              <Button type="submit" disabled={formik.isSubmitting}>
                {formik.isSubmitting ? '建立中...' : '新增'}
              </Button>
            </ButtonGroup>
          </Box>

          {/*

          */}
        </Box>
      </Modal>
    </div>
  );
}
