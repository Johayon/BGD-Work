import java.util.ArrayList;
import java.util.HashMap;


public class shepherd {
	
	private HashMap<Integer,ArrayList<String>> l_HostCmd;
	private ArrayList<String> ListeMachineUsable;
	
	public shepherd (HashMap<Integer,ArrayList<String>> HostCmd){
		l_HostCmd = HostCmd;
		ListeMachineUsable = new ArrayList<String>();
	}	
	
	public void exec() throws InterruptedException{
		ArrayList<MasterCommand> Threader = new ArrayList<MasterCommand>();
		
		 for(Integer Thread : l_HostCmd.keySet()){
	    	MasterCommand thread = new MasterCommand(l_HostCmd.get(Thread).get(0),l_HostCmd.get(Thread).get(1));
	    	Threader.add(thread);
	    	thread.start();
	      		}
		
	    for(MasterCommand thread: Threader){
	    	thread.join();
	    	}
	    
	    for(MasterCommand thread: Threader){
	    	if(thread.alive()){
	    		ListeMachineUsable.add(thread.hostname());
	    	}
	    }	
	}
	
	
	public ArrayList<String> Usable(){
		return ListeMachineUsable;
	}
	
}
