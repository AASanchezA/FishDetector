



// Copy of cv_mouse from cv_utilities
class Mouse
{
public:
	static void start(const std::string& a_img_name)
	{
		cvSetMouseCallback(a_img_name.c_str(), Mouse::cv_on_mouse, 0);
	}
	static int event(void)
	{
		int l_event = m_event;
		m_event = -1;
		return l_event;
	}
	static int x(void)
	{
		return m_x;
	}
	static int y(void)
	{
		return m_y;
	}

private:
	static void cv_on_mouse(int a_event, int a_x, int a_y, int, void *)
	{
		m_event = a_event;
		m_x = a_x;
		m_y = a_y;
	}

	static int m_event;
	static int m_x;
	static int m_y;
};
