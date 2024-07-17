import { Box,Avatar,Typography,Button,IconButton } from "@mui/material";
import { red } from "@mui/material/colors";
import { useAuth } from "../context/AuthContext";
import { IoMdSend } from "react-icons/io";
import { useRef,useState ,useEffect,useLayoutEffect} from "react";
import {  useNavigate } from "react-router-dom";
import ChatItem from "../components/chat/ChatItem";
import {
  sendChatRequest,
  deleteUserChats,
  getUserChats,
} from "../helper/api-communicator"
import toast from "react-hot-toast";
type Message = {
  role: "user" | "assistant";
  parts: string;
};
const Chat = () => {
  const navigate=useNavigate();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const auth = useAuth();
  
  const [chatMessages, setChatMessages] = useState<Message[]>([]);
  const handleSubmit = async () => {
    console.log(inputRef.current?.value);
    const parts = inputRef.current?.value as string;
    if (inputRef && inputRef.current) {
      inputRef.current.value = "";
    }
    const newMessage: Message = { role: "user", parts };
    setChatMessages((prev) => [...prev, newMessage]);
    const chatData = await sendChatRequest(parts);
    setChatMessages([...chatData.chats]);
  };

  //delete
  const handleDeleteChats = async () => {
    try {
      toast.loading("Deleting Chats", { id: "deletechats" });
      await deleteUserChats();
      setChatMessages([]);
      toast.success("Deleted Chats Successfully", { id: "deletechats" });
    } catch (error) {
      console.log(error);
      toast.error("Deleting chats failed", { id: "deletechats" });
    }
  };

  //get chat
  useLayoutEffect(() => {
    if (auth?.isLoggedIn && auth.user) {
      toast.loading("Loading Chats", { id: "loadchats" });
      getUserChats()
        .then((data) => {
          setChatMessages([...data.chats]);
          toast.success("Successfully loaded chats", { id: "loadchats" });
        })
        .catch((err) => {
          console.log(err);
          toast.error("Loading Failed", { id: "loadchats" });
        });
    }
  }, [auth]);

 useEffect(()=>{
  if(!auth?.user){
    return navigate("/login");
  }
 },[auth, navigate]);

  return (
    <Box   sx={{
      display: "flex",
      flex: 1,
      width: "100%",
      height: "100%",
      mt: 3,
      gap: 3,
      
    }}>
   <Box 
   sx={{
    display: { md: "flex", xs: "none", sm: "none" },
    flex: 0.2,
    flexDirection: "column",
  }}>
   <Box  sx={{
            display: "flex",
            width: "100%",
            height: "60vh",
            bgcolor: "rgb(17,29,39)",
            borderRadius: 5,
            flexDirection: "column",
            mx: 3,
          }}>
 <Avatar sx={{
   mx: "auto",
   my: 2,
   bgcolor: "white",
   color: "black",
   fontWeight: 700,
 }}>
{auth?.user?.name[0]}
{auth?.user?.name.split(" ")[1][0]}
 </Avatar>
 <Typography sx={{ mx: "auto", fontFamily: "work sans" }}>
            You are talking to a ChatBOT
          </Typography>
          <Typography sx={{ mx: "auto", fontFamily: "work sans", my: 4, p: 3 }}>
            You can ask some questions related to Knowledge, Business, Advices,
            Education, etc. But avoid sharing personal information
          </Typography>
          <Button 
            onClick={handleDeleteChats}
          sx={{
            width: "200px",
            my: "auto",
            color: "white",
            fontWeight: "700",
            borderRadius: 10,
            mx: "auto",
            bgcolor: red[300],
            ":hover": {
              bgcolor: red.A400,
            },
          }}
          >
       Clear Conversation
          </Button>
    </Box>
   </Box>

   <Box 
   sx={{
    display: "flex",
    flex: { md: 0.8, xs: 1, sm: 1 },
    flexDirection: "column",
    px: 3,
    overflowX: "hidden",
    wordWrap: "break-word",
    whiteSpace: "pre-wrap",
  }}
   >
<Typography
          sx={{
            fontSize: "40px",
            color: "white",
            mb: 2,
            mx: "auto",
            fontWeight: "600",
          }}
        >
          Gemini-1.5-flash
        </Typography>
        <Box sx={{
            width: "100%",
            height: "60vh",
            borderRadius: 3,
            mx: "auto",
            display: "flex",
            flexDirection: "column",
            overflow: "scroll",
            overflowX: "hidden",
            overflowY: "auto",
            scrollBehavior: "smooth",
          }}>
             {chatMessages.map((chat, index) => (
            <ChatItem parts={chat.parts} role={chat.role} key={index} />
          ))}


        </Box>
        <div
          style={{
            width: "100%",
            borderRadius: 80,
            backgroundColor: "rgb(17,27,39)",
            display: "flex",
            margin: "auto",
            
            wordWrap: "break-word",
            whiteSpace: "pre-wrap",
          }}
        >
          {" "}
        <input
             ref={inputRef}
            type="text"
            style={{
              width: "100%",
              backgroundColor: "transparent",
              padding: "25px",
              border: "none",
              outline: "none",
              color: "white",
              fontSize: "20px",
            }}
          />
            <IconButton onClick={handleSubmit} sx={{ color: "white", mx: 1 }}>
            <IoMdSend />
          </IconButton>
          </div>
    </Box>
    </Box>
  );
}

export default Chat;
