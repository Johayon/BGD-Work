import java.io.IOException;



public class MasterCommand extends Thread {

	private String l_host;
	private String l_cmd;
	private boolean alive;
	private String output;
	
	public MasterCommand (String hostname, String cmd){
		l_host=hostname;
		l_cmd=cmd;
		output="";
	}
	
	public void run(){  
		try {Hostssh2 local = new Hostssh2 (l_host);
				local.exec(l_cmd);
				/* System.out.println(finish); */
				alive = true;
				output = local.out();
		} 
		catch (IOException e) {
			/* System.out.println("error on " + l_host); */
			alive =false;
		}
	}
	public boolean alive(){
		return alive;
	}
	public String hostname(){
		return l_host;
	}
	public String out(){
		return output;
	}
	
		
}   

