import { memo ,useState} from 'react';
import { useRecoilValue } from 'recoil';

import { Box, Stack } from '@mui/material';
import useMediaQuery from '@mui/material/useMediaQuery';

import UserButton from 'components/atoms/buttons/userButton';
import { Logo } from 'components/atoms/logo';
import ChatProfiles from 'components/molecules/chatProfiles';
import NewChatButton from 'components/molecules/newChatButton';

import { settingsState } from 'state/settings';

import { OpenSideBarMobileButton } from './sidebar/OpenSideBarMobileButton';
import AddIcon from '@mui/icons-material/Add';
import { SettingsModal } from 'components/SettingsModal'
import { AgentModalInfo } from 'components/AgentModalInfo'
import IconButton from '@mui/material/IconButton';

const Header = memo(() => {
  const isMobile = useMediaQuery('(max-width: 66rem)');
  const { isChatHistoryOpen } = useRecoilValue(settingsState);
  const [isSettingsModalOpen, setSettingsModalOpen] = useState(false);
  return (
    <>
      <AgentModalInfo
        isOpen={isSettingsModalOpen}
        onClose={() => setSettingsModalOpen(false)}
      />
      <Box
        px={1}
        py={1}
        display="flex"
        height="45px"
        alignItems="center"
        flexDirection="row"
        justifyContent="space-between"
        color="text.primary"
        gap={2}
        id="header"
        position="relative"
      >
        <Stack direction="row" alignItems="center" gap={1.5}>
          {isMobile ? (
            <OpenSideBarMobileButton />
          ) : isChatHistoryOpen ? null : (
            <Logo style={{ maxHeight: '25px' }} />
          )}
          <IconButton
            aria-label="add"
            onClick={() => setSettingsModalOpen(true)}
            sx={{
              border: '1px solid',
              borderColor: 'grey.300', // 給一個柔和的邊框色
              borderRadius: '50%',
              width: 32,
              height: 32,
            }}
          >
            <AddIcon fontSize="small" />
          </IconButton>
        </Stack>
        <Box>
          <ChatProfiles />
        </Box>
        <Stack direction="row" alignItems="center">
          <NewChatButton />
          <UserButton />
        </Stack>
      </Box>
    </>
  );
});

export { Header };
