import React from 'react';
import { useFormik } from 'formik';
import * as yup from 'yup';
import { toast } from 'sonner';

// Material-UI Components
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import Box from '@mui/material/Box';

// 定義 Modal 的 Props 型別
interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

// 修正後的驗證規則，確保與表單欄位完全對應
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

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
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
    // 更新後的 onSubmit 邏輯，串接後端 API
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

  // 對於「新增」功能的 Modal，通常不需要從 localStorage 載入舊資料，
  // 因此移除了 useEffect hook，確保每次打開都是一個乾淨的表單。

  return (
    <Dialog open={isOpen} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>新增 Agent</DialogTitle>
      <Box component="form" onSubmit={formik.handleSubmit} noValidate>
        <DialogContent>
          {/* Agent Name */}
          <TextField
            autoFocus
            margin="dense"
            id="agentName"
            name="agentName"
            label="Agent Name"
            type="text"
            fullWidth
            variant="outlined"
            required
            value={formik.values.agentName}
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
            error={formik.touched.agentName && Boolean(formik.errors.agentName)}
            helperText={formik.touched.agentName && formik.errors.agentName}
          />

          {/* LLM Model Name */}
          <TextField
            margin="dense"
            id="llmModelName"
            name="llmModelName"
            label="LLM Model Name"
            type="text"
            fullWidth
            variant="outlined"
            required
            value={formik.values.llmModelName}
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
            error={formik.touched.llmModelName && Boolean(formik.errors.llmModelName)}
            helperText={formik.touched.llmModelName && formik.errors.llmModelName}
          />

          {/* API Endpoint */}
          <TextField
            margin="dense"
            id="endpoint"
            name="endpoint"
            label="API Endpoint"
            type="text"
            fullWidth
            variant="outlined"
            required
            value={formik.values.endpoint}
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
            error={formik.touched.endpoint && Boolean(formik.errors.endpoint)}
            helperText={formik.touched.endpoint && formik.errors.endpoint}
          />

          {/* LLM API Key */}
          <TextField
            margin="dense"
            id="llmApiKey"
            name="llmApiKey"
            label="LLM API Key"
            type="password"
            fullWidth
            variant="outlined"
            required
            value={formik.values.llmApiKey}
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
            error={formik.touched.llmApiKey && Boolean(formik.errors.llmApiKey)}
            helperText={formik.touched.llmApiKey && formik.errors.llmApiKey}
          />

          {/* Embedding Model Name */}
          <TextField
            margin="dense"
            id="embeddingModalName"
            name="embeddingModalName"
            label="Embedding Model Name"
            type="text"
            fullWidth
            variant="outlined"
            required
            value={formik.values.embeddingModalName}
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
            error={formik.touched.embeddingModalName && Boolean(formik.errors.embeddingModalName)}
            helperText={formik.touched.embeddingModalName && formik.errors.embeddingModalName}
          />

          {/* Embedding Endpoint */}
          <TextField
            margin="dense"
            id="embeddingEndpoint"
            name="embeddingEndpoint"
            label="Embedding Endpoint"
            type="text"
            fullWidth
            variant="outlined"
            required
            value={formik.values.embeddingEndpoint}
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
            error={formik.touched.embeddingEndpoint && Boolean(formik.errors.embeddingEndpoint)}
            helperText={formik.touched.embeddingEndpoint && formik.errors.embeddingEndpoint}
          />

          {/* Embedding Api Key */}
          <TextField
            margin="dense"
            id="embeddingApiKey"
            name="embeddingApiKey"
            label="Embedding Api Key"
            type="password"
            fullWidth
            variant="outlined"
            required
            value={formik.values.embeddingApiKey}
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
            error={formik.touched.embeddingApiKey && Boolean(formik.errors.embeddingApiKey)}
            helperText={formik.touched.embeddingApiKey && formik.errors.embeddingApiKey}
          />

          {/* DB Connection String */}
          <TextField
            margin="dense"
            id="dbConnectionString"
            name="dbConnectionString"
            label="DB Connection String"
            type="password"
            fullWidth
            variant="outlined"
            required
            value={formik.values.dbConnectionString}
            onChange={formik.handleChange}
            onBlur={formik.handleBlur}
            error={formik.touched.dbConnectionString && Boolean(formik.errors.dbConnectionString)}
            helperText={formik.touched.dbConnectionString && formik.errors.dbConnectionString}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose}>取消</Button>
          <Button type="submit" disabled={formik.isSubmitting}>
            {formik.isSubmitting ? '儲存中...' : '儲存'}
          </Button>
        </DialogActions>
      </Box>
    </Dialog>
  );
};
