import { Box,Avatar,Typography } from "@mui/material";
import { useAuth } from "../../context/AuthContext.js";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { coldarkDark } from 'react-syntax-highlighter/dist/esm/styles/prism';


function extractCodeFromString(message: string) {
    const blocks = [];
    if (message.includes("```")) {
      const parts = message.split("```");
      for (let i = 0; i < parts.length; i++) {
        if (i % 2 === 0) {
          blocks.push({ text: parts[i], language: "" });
        } else {
          const [language, ...code] = parts[i].split("\n");
          blocks.push({ text: code.join("\n"), language });
        }
      }
    } else {
      blocks.push({ text: message, language: "" });
    }
    return blocks;
  }
  
  function isCodeBlock(str: string) {
    if (
      str.includes("=") ||
      str.includes(";") ||
      str.includes("[") ||
      str.includes("]") ||
      str.includes("{") ||
      str.includes("}") ||
      str.includes("#") ||
      str.includes("//")
    ) {
      return true;
    }
    return false;
  }
const ChatItem = ({
   parts,
    role,
}:{parts:string,
role:"user" | "assistant";
}) => {
    const messageBlocks = extractCodeFromString(parts);
    const auth=useAuth();
  return role==="user"? ( <Box
  sx={{
    display: "flex",
    p: 2,
    bgcolor: "#004d5612",
    gap: 2,
    borderRadius: 2,
    my: 1,
    wordWrap: "break-word",
    whiteSpace: "pre-wrap",
  }}>
    <Avatar sx={{ ml: "0" }}>
    
    
    {auth?.user?.name[0]}
    {auth?.user?.name.split(" ")[1][0]}
      </Avatar>
      <Box>
        {/* <Typography fontSize={"20px"}>{parts}</Typography> */}
        {!messageBlocks && (
          <Typography sx={{ fontSize: "20px" }}>{parts}</Typography>
        )}
        {messageBlocks &&
          messageBlocks.length &&
          messageBlocks.map((block) =>
            isCodeBlock(block.text) ? (
              <SyntaxHighlighter style={coldarkDark} language={block.language}>
                {block.text}
              </SyntaxHighlighter>
            ) : (
              <Typography sx={{ fontSize: "20px" }}>{block.text}</Typography>
            )
          )}
      </Box>
  </Box>)
  :
  (<Box
  sx={{
    display: "flex",
    p: 2,
    bgcolor: "#004d56",
    gap: 2,
    my:2,
    borderRadius: 2,
    wordWrap: "break-word",
    whiteSpace: "pre-wrap",
  }}>
    <Avatar  sx={{ ml: "0", bgcolor: "black", color: "white" }}>
    
    <img src="airobot.png" alt="GEMINI" width={"30px"} />
    
      </Avatar>
      <Box>
      {!messageBlocks && (
          <Typography sx={{ fontSize: "20px" }}>{parts}</Typography>
        )}
        {messageBlocks &&
          messageBlocks.length &&
          messageBlocks.map((block) =>
            isCodeBlock(block.text) ? (
              <SyntaxHighlighter style={coldarkDark} language={block.language}>
                {block.text}
              </SyntaxHighlighter>
            ) : (
              <Typography sx={{ fontSize: "20px" }}>{block.text}</Typography>
            )
          )}
        {/* <Typography fontSize={"20px"}>{parts}</Typography> */}
      </Box>
  </Box>);
  };
  

export default ChatItem;